import logging
from datetime import datetime

from src.utils.paths import output_path


def setup_logging(config, name):
    path = output_path(config, f'logs/{name}_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding='utf-8')], force=True)
    return path
