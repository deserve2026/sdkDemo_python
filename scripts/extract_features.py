import sys
sys.dont_write_bytecode = True
from _entry import PROJECT_ROOT
import argparse
from pathlib import Path
from src.data.features import extract
from src.utils.experiment import read_settings
from src.utils.io import load_config
from src.utils.logging import setup_logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache frozen ResNet18 features (train/val only)')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'dataset.yaml')
    parser.add_argument('--model-config', type=Path, default=PROJECT_ROOT / 'configs' / 'model.yaml')
    parser.add_argument('--train-config', type=Path, default=PROJECT_ROOT / 'configs' / 'train.yaml')
    parser.add_argument('--splits', nargs='+', choices=('train', 'val'), default=['train', 'val'])
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config, 'extract_features')
    extract(config, read_settings(args.model_config), read_settings(args.train_config), args.splits)
