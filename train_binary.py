"""Run the unmodified VAR trainer with two-class folder datasets.

Pass VAR's normal training arguments through this entry point. Only the dataset
builder is replaced, so optimization, logging, checkpoints and resume remain
the upstream implementation.
"""

import argparse
import runpy
import sys
from pathlib import Path

from datasets import build_binary_training_datasets


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
    sys.argv = [str(train_script), *upstream_args]
    runpy.run_path(str(train_script), run_name='__main__')


if __name__ == '__main__':
    main()
