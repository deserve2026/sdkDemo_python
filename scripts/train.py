import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from _entry import PROJECT_ROOT
import argparse
from pathlib import Path
from src.training.trainer import train
from src.utils.experiment import read_settings
from src.utils.io import load_config
from src.utils.logging import setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train small TCN and select on validation only')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'dataset.yaml')
    parser.add_argument('--model-config', type=Path, default=PROJECT_ROOT / 'configs' / 'model.yaml')
    parser.add_argument('--train-config', type=Path, default=PROJECT_ROOT / 'configs' / 'train.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config, 'train')
    train(config, read_settings(args.model_config), read_settings(args.train_config), PROJECT_ROOT)

