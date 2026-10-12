"""Drive the version 2 cross-validation grid. Cached jobs are reused."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/'.venv/Scripts/python.exe'
YEARS = [2021, 2022, 2023, 2024]

GRID = {
    # (model, objective, features, season, since)
    'lightgbm': [
        ('lightgbm', 'delta', 'v1', 'all', 2017),            # control: must match version 1
        ('lightgbm', 'delta', 'full2', 'all', 2017),
        ('lightgbm', 'delta', 'full2', 'all', 2012),
        ('lightgbm', 'delta', 'v1_thermal', 'all', 2017),
        ('lightgbm', 'delta', 'v1_hydro', 'all', 2017),
        ('lightgbm', 'delta', 'v1_algaewater', 'all', 2017),
    ],
    'tabpfn': [
        ('tabpfn', 'delta', 'full2', 'all', 2012),
        ('tabpfn', 'delta', 'v1_thermal', 'all', 2012),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['lightgbm'])
    parser.add_argument('--years', nargs='+', type=int, default=YEARS)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    jobs = [job for model in args.models for job in GRID[model]]
    for model, objective, features, season, since in jobs:
        for year in args.years:
            command = [str(PYTHON), '-m', 'algae.exp2', 'run', '--model', model,
                       '--objective', objective, '--features', features, '--season', season,
                       '--since', str(since), '--year', str(year)]
            result = subprocess.run(command, cwd=ROOT, timeout=args.timeout)
            if result.returncode:
                sys.exit(f'FAILED: {command}')


if __name__ == '__main__':
    main()
