import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, mark_stage, write_json
from src.utils.logging import setup_logging

MODULES = {
    'audit': 'src.data.audit',
    'timeline': 'src.data.timeline',
    'events': 'src.data.events',
    'split': 'src.data.splitting',
    'windows': 'src.data.windowing',
}


def main(stage):
    parser = argparse.ArgumentParser(description='Read-only sonar source-data pipeline')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'dataset.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config, stage)
    stages = list(MODULES) if stage == 'pipeline' else [stage]
    if stage == 'pipeline':
        write_json(config, 'metadata/pipeline_status.json', {
            'config_sha256': config['_config_sha256'], 'stages': {},
            'data_ready_for_training': False, 'training_started': False,
        })
    for name in stages:
        logging.info('Starting stage: %s', name)
        try:
            result = importlib.import_module(MODULES[name]).run(config)
        except Exception as error:
            logging.exception('Stage failed: %s', name)
            # Existing reports remain available; downstream stages cannot use stale success.
            mark_stage(config, name, False, {'error': str(error)})
            raise SystemExit(1) from error
        logging.info('Completed %s: %s', name, result)
    logging.info('Finished. No model training has been started.')


def deferred(name):
    raise SystemExit(
        f'{name} is intentionally not implemented in the data-pipeline delivery. '
        'Finish and review reports first. No training, evaluation, or ONNX export is performed.'
    )

