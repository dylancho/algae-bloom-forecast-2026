"""Choose one tree family on 2021-2024 CV, then test bounded feature/history changes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
groups={}
for path in (ROOT/'artifacts/cv').glob('*/metrics.json'):
    m=json.loads(path.read_text())
    if m['tag'].startswith(('catboost_','lightgbm_')) and m['tag'].endswith('_full_all_2012'):
        groups.setdefault(m['tag'],[]).append(m)
complete={k:v for k,v in groups.items() if {x['year'] for x in v}=={2021,2022,2023,2024}}
best=min(complete,key=lambda k:sum(x['season']['rmse_log10'] for x in complete[k])/4)
model,objective=best.split('_')[:2]
choice={'base_candidate':best,'selection':'Mean 2021-2024 May-Oct log10 RMSE; no 2025 used.',
        'variants':[{'features':f,'season':s,'since':y} for f,s,y in
                    [('algae','all',2012),('water','all',2012),('weather','all',2012),
                     ('full','warm',2012),('full','all',2017)]]}
(ROOT/'artifacts/ablation_plan.json').write_text(json.dumps(choice,indent=2),encoding='utf-8')
print('ABLATION BASE',best,flush=True)
for variant in choice['variants']:
    for year in [2021,2022,2023,2024]:
        cmd=[sys.executable,'-m','algae.experiment','run','--model',model,'--objective',objective,
             '--year',str(year),'--features',variant['features'],'--season',variant['season'],
             '--since',str(variant['since'])]
        subprocess.run(cmd,cwd=ROOT,check=True,timeout=900)
subprocess.run([sys.executable,'-m','algae.experiment','summarize'],cwd=ROOT,check=True)
