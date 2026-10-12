"""Seven-panel A/U report for the binary VAR likelihood classifier."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from au_visualization import write_report


def _read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'sample_id', 'label'} <= set(reader.fieldnames or []):
            raise ValueError('manifest needs sample_id,label columns')
        rows = []
        seen = set()
        for row in reader:
            sample_id = row['sample_id']
            if not sample_id or sample_id != sample_id.strip() or sample_id in seen:
                raise ValueError(f'empty, whitespace-padded or duplicate sample_id: {sample_id!r}')
            if row['label'] not in ('0', '1'):
                raise ValueError(f'invalid label for {sample_id}: {row["label"]!r}')
            seen.add(sample_id)
            rows.append((sample_id, int(row['label'])))
    if not rows:
        raise ValueError('manifest is empty')
    return rows


def _select_samples(dataset, data_path, split, manifest, max_samples, seed):
    root = Path(data_path).resolve()
    lookup = {}
    for index, (path, label) in enumerate(dataset.samples):
        relative = Path(path).resolve().relative_to(root).as_posix()
        lookup[relative] = (index, label)
        lookup[relative.removeprefix(f'{split}/')] = (index, label)
    if manifest:
        selected = []
        for sample_id, expected_label in _read_manifest(manifest):
            key = sample_id.replace('\\', '/')
            if key not in lookup:
                raise ValueError(f'manifest sample not found in {split}: {sample_id}')
            index, actual_label = lookup[key]
            if expected_label != actual_label:
                raise ValueError(f'label conflict for {sample_id}: manifest={expected_label}, dataset={actual_label}')
            selected.append((index, sample_id, actual_label))
        if len({index for index, _, _ in selected}) != len(selected):
            raise ValueError('manifest maps multiple IDs to the same dataset sample')
        return selected
    if max_samples < 1:
        raise ValueError('--max_samples must be >= 1')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    return [(int(i), Path(dataset.samples[int(i)][0]).resolve().relative_to(root).as_posix(),
             int(dataset.samples[int(i)][1])) for i in indices]


def _file_record(path):
    p = Path(path).resolve()
    stat = p.stat()
    return {'path': str(p), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _extract(args):
    import torch
    from torch.utils.data import DataLoader, Subset

    from datasets import build_dataset
    from infer_binary import load_models

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('--batch_size must be >= 1 and --num_workers must be >= 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32) if args.depth == 36 else (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    final_reso = 512 if args.depth == 36 else 256
    dataset = build_dataset(data_path=args.data_path, final_reso=final_reso,
                            dataset_type='binary', split=args.split)
    selected = _select_samples(dataset, args.data_path, args.split,
                               args.sample_manifest, args.max_samples, args.seed)
    if len(selected) < 2:
        raise ValueError('at least two selected samples are needed for the seven-panel figure')
    args.model_ckpt = args.checkpoint  # Reuse the existing strict checkpoint loader.
    vae, var_model = load_models(args, device, patch_nums)
    if not isinstance(getattr(var_model, 'head', None), torch.nn.Module):
        raise RuntimeError('VAR final token head module was not found; inspect this checkpoint architecture')
    loader = DataLoader(Subset(dataset, [i for i, _, _ in selected]), batch_size=args.batch_size,
                        shuffle=False, drop_last=False, num_workers=args.num_workers,
                        pin_memory=device.type == 'cuda')
    rows = []
    seen_labels = []
    # Hook the input of the real token-prediction Linear head, after the model's
    # own conditioning/normalization. One hook call must occur per class forward.
    captures = []
    token_shapes = set()

    def capture_head_input(_module, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 3:
            raise RuntimeError('expected VAR head input [B,L,C]')
        token_shapes.add(tuple(int(v) for v in inputs[0].shape[1:]))
        if len(token_shapes) != 1:
            raise RuntimeError('VAR token representation shape changed between forwards')
        captures.append(inputs[0].detach().float().mean(dim=1).cpu().numpy())

    hook = var_model.head.register_forward_pre_hook(capture_head_input)
    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                post = vae.img_to_post(images)
                token_maps = vae.quantize.f_to_idxBl_or_fhat(post, to_fhat=False, v_patch_nums=None)
                var_inputs = vae.quantize.idxBl_to_var_input(token_maps)
                batch_representations = []
                for class_index in (0, 1):
                    captures.clear()
                    conditions = torch.full((len(images),), class_index, dtype=torch.long, device=device)
                    var_model.forward(conditions, var_inputs)
                    if len(captures) != 1 or captures[0].shape[0] != len(images):
                        raise RuntimeError('VAR head hook did not capture exactly one vector per sample and class')
                    batch_representations.append(captures.pop())
                rows.append(np.concatenate(batch_representations, axis=1))
                seen_labels.extend(labels.tolist())
    finally:
        hook.remove()
    features = np.concatenate(rows, axis=0)
    labels = np.asarray(seen_labels, dtype=np.int64)
    expected = np.asarray([label for _, _, label in selected], dtype=np.int64)
    if len(features) != len(selected) or not np.array_equal(labels, expected):
        raise RuntimeError('extracted feature rows do not match selected sample IDs and labels')
    if not np.isfinite(features).all():
        raise RuntimeError('non-finite VAR head representation')
    token_count, token_width = next(iter(token_shapes))
    metadata = {
        'model': f'Binary VAR-d{args.depth}',
        'representation': 'var_head_input_token_mean_both_conditions',
        'representation_detail': 'For each image, run the actual full-scale VAR forward under benign and mal conditions. Capture the input [B,L,C] to var_model.head after its own conditioning and normalization; mean over L separately, then concatenate [benign_C, mal_C]. This is a diagnostic summary of both conditional token representations, not the classifier likelihood or a native single-vector readout.',
        'token_count': token_count,
        'token_width': token_width,
        'checkpoint': _file_record(args.checkpoint),
        'vae_checkpoint': _file_record(args.vae_ckpt),
        'split': args.split,
        'data_path': str(Path(args.data_path).resolve()),
        'source_manifest': str(Path(args.sample_manifest).resolve()) if args.sample_manifest else None,
        'training_seed': None,
        'sampling_seed': None if args.sample_manifest else args.seed,
        'max_samples': None if args.sample_manifest else args.max_samples,
        'preprocessing': f'build_dataset(binary): Resize({round(1.125 * final_reso)}, LANCZOS), CenterCrop({final_reso}), ToTensor, map [0,1] to [-1,1]',
        'device': str(device),
    }
    return features, labels, np.asarray([sample_id for _, sample_id, _ in selected], dtype=str), metadata


def main():
    parser = argparse.ArgumentParser(description='A/U seven-panel report for binary VAR')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', '--model_ckpt', help='Trained binary VAR checkpoint')
    source.add_argument('--features_npz', help='Previously exported features.npz (no model required)')
    parser.add_argument('--data_path', help='Parent directory of split/benign and split/mal')
    parser.add_argument('--split', default='val')
    parser.add_argument('--sample_manifest', help='CSV with sample_id,label; exact selection and order')
    parser.add_argument('--max_samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--vae_ckpt', default='vae_ch160v4096z32.pth')
    parser.add_argument('--depth', type=int, default=16, choices=[16, 20, 24, 30, 36])
    parser.add_argument('--batch_size', type=int, default=1, help='Image batch size')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--align_alpha', type=float, default=2.0)
    parser.add_argument('--uniform_t', type=float, default=2.0)
    args = parser.parse_args()
    if args.features_npz:
        if args.sample_manifest or args.data_path:
            parser.error('--features_npz uses every cached row; do not provide data_path or sample_manifest')
        with np.load(args.features_npz, allow_pickle=False) as cached:
            if set(cached.files) != {'features', 'labels', 'sample_ids'}:
                parser.error('features NPZ must contain exactly features, labels, sample_ids')
            features, labels, ids = cached['features'], cached['labels'], cached['sample_ids']
        previous = Path(args.features_npz).parent / 'metadata.json'
        metadata = json.loads(previous.read_text(encoding='utf-8')) if previous.is_file() else {
            'model': 'Cached features', 'representation': 'unknown', 'training_seed': None,
            'sampling_seed': None, 'split': None,
        }
        metadata['redrawn_from'] = str(Path(args.features_npz).resolve())
    else:
        if not args.data_path:
            parser.error('--data_path is required with --checkpoint')
        features, labels, ids, metadata = _extract(args)
    metrics = write_report(args.output_dir, features, labels, ids, metadata,
                           align_alpha=args.align_alpha, uniform_t=args.uniform_t)
    print(json.dumps({'output_dir': str(Path(args.output_dir).resolve()), 'metrics': metrics}, indent=2))
    if metrics['zero_vector_count']:
        print('WARNING: zero vectors have no sphere direction; retained in high-D metrics by the stated normalization convention')


if __name__ == '__main__':
    main()
