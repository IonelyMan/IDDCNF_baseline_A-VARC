"""Evaluate a trained two-class VAR checkpoint on a labeled image split."""

import argparse
import math
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


CLASS_NAMES = ('benign', 'mal')


def calculate_metrics(y_true, y_pred, mal_scores):
    """Return binary metrics; mal (label 1) is the positive class."""
    if len(y_true) != len(y_pred) or len(y_true) != len(mal_scores):
        raise ValueError('Labels, predictions and scores must have equal lengths')
    if set(y_true) != {0, 1}:
        raise ValueError('ROC AUC and PR AUC require both benign and mal samples')

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision, recall, _ = precision_recall_curve(y_true, mal_scores, pos_label=1)

    metrics = {
        'ACC': accuracy_score(y_true, y_pred),
        'PR AUC': auc(recall, precision),
        'ROC AUC': roc_auc_score(y_true, mal_scores),
        'ACSA': (sensitivity + specificity) / 2,
        'GM': math.sqrt(sensitivity * specificity),
        'RECALL': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    )
    return metrics, matrix, report


def load_models(args, device, patch_nums):
    import torch

    for checkpoint in (args.vae_ckpt, args.model_ckpt):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    if not (Path(__file__).resolve().parent / 'VAR' / 'models' / '__init__.py').is_file():
        raise FileNotFoundError('VAR submodule is missing; run git submodule update --init --recursive')

    from VAR_extensions import build_vae_var

    vae, var_model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums, num_classes=2,
        depth=args.depth, shared_aln=args.depth == 36,
    )
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu', weights_only=False), strict=True)

    checkpoint = torch.load(args.model_ckpt, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'trainer' in checkpoint:
        checkpoint = checkpoint['trainer']
    if isinstance(checkpoint, dict) and 'var_wo_ddp' in checkpoint:
        checkpoint = checkpoint['var_wo_ddp']
    var_model.load_state_dict(checkpoint, strict=True)

    vae.eval()
    var_model.eval()
    var_model.cond_drop_rate = 0
    return vae, var_model


def run_inference(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from datasets import build_dataset

    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32) if args.depth == 36 else (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    final_reso = 512 if args.depth == 36 else 256
    dataset = build_dataset(
        data_path=args.data_path, final_reso=final_reso,
        dataset_type='binary', split=args.split,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    vae, var_model = load_models(args, device, patch_nums)
    # As in eval.py, batch_size limits model forward combinations, not images.
    image_batch_size = max(args.batch_size // 2, 1)
    loader = DataLoader(
        dataset, batch_size=image_batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
    )

    y_true, y_pred, mal_scores = [], [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f'Inferring {args.split}'):
            images = images.to(device, non_blocking=True)
            features = vae.img_to_post(images)
            token_maps = vae.quantize.f_to_idxBl_or_fhat(features, to_fhat=False, v_patch_nums=None)
            gt_tokens = torch.cat(token_maps, dim=1)
            var_inputs = vae.quantize.idxBl_to_var_input(token_maps)

            class_scores = []
            for class_index in (0, 1):
                conditions = torch.full((images.shape[0],), class_index, dtype=torch.long, device=device)
                logits = var_model.forward(conditions, var_inputs)
                token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
                    dim=-1, index=gt_tokens.unsqueeze(-1)
                ).squeeze(-1)
                class_scores.append(token_log_probs.sum(dim=-1))

            likelihoods = torch.stack(class_scores, dim=1)
            if not torch.isfinite(likelihoods).all():
                raise RuntimeError('Non-finite VAR log-likelihood encountered')
            predictions = likelihoods.argmax(dim=1)
            margins = likelihoods[:, 1] - likelihoods[:, 0]
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())
            mal_scores.extend(margins.cpu().tolist())

    return calculate_metrics(y_true, y_pred, mal_scores)


def main():
    parser = argparse.ArgumentParser(description='Two-class VAR inference and metrics')
    parser.add_argument('--data_path', required=True, help='Parent directory containing split/benign and split/mal')
    parser.add_argument('--split', default='test', help='Labeled split to evaluate')
    parser.add_argument('--model_ckpt', required=True, help='Trained two-class VAR checkpoint')
    parser.add_argument('--vae_ckpt', default='vae_ch160v4096z32.pth', help='Pretrained VQ-VAE checkpoint')
    parser.add_argument('--depth', type=int, default=16, choices=[16, 20, 24, 30, 36])
    parser.add_argument('--batch_size', type=int, default=1, help='Maximum model forward batch size')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch_size must be >= 1')
    if args.num_workers < 0:
        parser.error('--num_workers must be >= 0')

    metrics, matrix, report = run_inference(args)
    for name, value in metrics.items():
        print(f'{name}: {value:.6f}')
    print('\nConfusion matrix (rows=true, columns=predicted; order: benign, mal):')
    print(matrix)
    print('\nClassification report:')
    print(report)


if __name__ == '__main__':
    main()
