"""Run the upstream VAR trainer with two-class folder datasets.

Pass VAR's normal training arguments through this entry point. The dataset
builder is replaced, and a logging-only fix handles gradient accumulation.
Optimization, checkpoints and resume remain the upstream implementation.
"""

import argparse
import runpy
import sys
from pathlib import Path

from datasets import build_binary_training_datasets


def fix_gradient_accumulation_logging():
    """Give upstream trainer.py a scalar to log before the first optimizer step.

    AmpOptimizer returns ``None`` for the gradient norm on accumulation steps,
    while VARTrainer.train_step unconditionally calls ``grad_norm.item()`` when
    logging. A zero placeholder avoids that crash without changing gradients or
    the optimizer step. The actual norm is returned on optimizer steps.
    """
    from utils.amp_sc import AmpOptimizer

    original_backward_clip_step = AmpOptimizer.backward_clip_step

    def backward_clip_step(self, stepping, loss):
        grad_norm, scale_log2 = original_backward_clip_step(
            self, stepping=stepping, loss=loss,
        )
        if grad_norm is None:
            grad_norm = loss.detach().new_zeros(())
        return grad_norm, scale_log2

    AmpOptimizer.backward_clip_step = backward_clip_step


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--val_split', default='val')
    split_args, upstream_args = parser.parse_known_args()

    var_dir = Path(__file__).resolve().parent / 'VAR'
    train_script = var_dir / 'train.py'
    if not train_script.is_file():
        raise FileNotFoundError('VAR submodule is missing; run git submodule update --init --recursive')

    sys.path.insert(0, str(var_dir))
    from utils import data as var_data

    def binary_dataset(data_path, final_reso, hflip=False, mid_reso=1.125):
        return build_binary_training_datasets(
            data_path=data_path, train_split=split_args.train_split,
            val_split=split_args.val_split, final_reso=final_reso,
            hflip=hflip, mid_reso=mid_reso,
        )

    var_data.build_dataset = binary_dataset
    fix_gradient_accumulation_logging()
    sys.argv = [str(train_script), *upstream_args]
    runpy.run_path(str(train_script), run_name='__main__')


if __name__ == '__main__':
    main()
