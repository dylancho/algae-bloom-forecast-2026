"""Run independently cached historical CV jobs; never selects on 2025."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',default=['catboost','lightgbm'])
parser.add_argument('--years',nargs='+',type=int,default=[2021,2022,2023,2024])
parser.add_argument('--timeout',type=int,default=1800)
args=parser.parse_args()
for model in args.models:
    for objective in ['direct','delta']:
        for year in args.years:
            cmd=[sys.executable,'-m','algae.experiment','run','--model',model,'--objective',objective,'--year',str(year)]
            try:
                result=subprocess.run(cmd,cwd=ROOT,timeout=args.timeout)
                if result.returncode:
                    print('FAILED',model,objective,year,flush=True)
            except subprocess.TimeoutExpired:
                folder=ROOT/'artifacts/cv'/f'{year}_{model}_{objective}_full_all_2012'
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'error.txt').write_text(f'CPU runtime exceeded {args.timeout} seconds.',encoding='utf-8')
                print('TIMEOUT',model,objective,year,flush=True)
                if model=='tabpfn':
                    sys.exit(2)
subprocess.run([sys.executable,'-m','algae.experiment','summarize'],cwd=ROOT,check=True)
