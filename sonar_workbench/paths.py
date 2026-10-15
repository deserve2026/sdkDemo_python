import hashlib
from pathlib import Path

RESULTS = Path('F:/experimental_results/tcn/tcn_feed')
PROJECT = Path(__file__).resolve().parents[1]


def output_path(relative):
    target = (RESULTS / relative).resolve()
    if not target.is_relative_to(RESULTS.resolve()):
        raise ValueError('Output must stay under the sonar results directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
