"""Freeze a small candidate set on historical CV before opening 2025 performance."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from algae.experiment import ROOT,metrics


def collect():
    groups={}
    for path in (ROOT/'artifacts/cv').glob('*/metrics.json'):
        m=json.loads(path.read_text())
        if m['year'] not in [2021,2022,2023,2024]:
            continue
        groups.setdefault(m['tag'],{})[m['year']]=path.parent
    candidates={}
    for tag,folds in groups.items():
        if set(folds)!={2021,2022,2023,2024}:
            continue
        frames=[]
        for year,path in sorted(folds.items()):
            p=pd.read_csv(path/'predictions.csv',parse_dates=['base_date','target_date'])
            p['year']=year
            frames.append(p)
        p=pd.concat(frames).sort_values(['base_date','station_id']).reset_index(drop=True)
        config=json.loads((folds[2024]/'config.json').read_text())
        candidates[tag]={'data':p,'config':config}
    return candidates


def score(frame,z):
    return float(np.mean([metrics(g.target,np.asarray(z)[g.index])['rmse_log10']
                          for _,g in frame[frame.base_date.dt.month.between(5,10)].groupby('year')]))


def select():
    candidates=collect()
    if not candidates:
        raise RuntimeError('Need four complete historical folds.')
    ranking=[]
    for tag,entry in candidates.items():
        p=entry['data']
        ranking.append({'tag':tag,'mean_season_rmse':score(p,p.pred_log10)})
    ranking.sort(key=lambda x:x['mean_season_rmse'])
    # Only 3 strongest complete candidates; coarse pairwise blends, no stacking fit.
    top=[r['tag'] for r in ranking[:3]]
    reference=candidates[top[0]]['data']
    best={'weights':{top[0]:1.0},'cv_mean_season_rmse':ranking[0]['mean_season_rmse']}
    blends=[]
    for a,b in itertools.combinations(top,2):
        left,right=candidates[a]['data'],candidates[b]['data']
        pd.testing.assert_frame_equal(left[['station_id','base_date','target_date','target']],
                                      right[['station_id','base_date','target_date','target']])
        for weight in [.25,.5,.75]:
            z=weight*left.pred_log10.to_numpy()+(1-weight)*right.pred_log10.to_numpy()
            value=score(left,z)
            trial={'weights':{a:weight,b:1-weight},'cv_mean_season_rmse':value}
            blends.append(trial)
            if value < best['cv_mean_season_rmse']-.002:
                best=trial
    result={**best,'selection_metric':'Equal mean of 2021-2024 May-Oct log10 RMSE; provisional, not official combined score.',
            'holdout_year':2025,'ranking':ranking,'blend_trials':blends,
            'components':{tag:candidates[tag]['config'] for tag in best['weights']},
            'confirm_single_candidates':top,
            'all_configs':{tag:candidates[tag]['config'] for tag in top},
            'test_labels_used':False,'valid_labels_used_for_selection':False}
    target=ROOT/'artifacts/selection.json'
    if target.exists():
        raise RuntimeError('Selection is already frozen. Do not silently reselect after holdout evaluation.')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    z=sum(w*candidates[tag]['data'].pred_log10.to_numpy() for tag,w in best['weights'].items())
    oof=reference[['station_id','base_date','target_date','target','year','cells_log_lag0']].copy()
    oof['pred_log10']=z
    oof.to_csv(ROOT/'artifacts/selected_oof.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    select()
