"""Confirm a previously frozen selection, save models, and create a draft submission."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from algae.experiment import ROOT,design,get_data,metrics
from algae.features import load_bundle


def model_folder(tag):
    return ROOT/'artifacts/holdout'/f'2025_{tag}'


def predict_saved(weights,data):
    z=np.zeros(len(data))
    for tag,weight in weights.items():
        fitted=joblib.load(model_folder(tag)/'model.joblib')
        pred=fitted['model'].predict(design(data,fitted['features'],fitted['model_name']))
        if fitted['objective']=='delta':
            pred=pred+data.cells_log_lag0.to_numpy(float)
        z+=weight*np.maximum(pred,0)
    assert np.isfinite(z).all()
    return z


def main():
    frozen=ROOT/'artifacts/selection.json'
    selection=json.loads(frozen.read_text())
    selection_hash=hashlib.sha256(frozen.read_bytes()).hexdigest()
    outputs=ROOT/'results'
    outputs.mkdir(exist_ok=True)
    # Exactly the three candidates frozen before any 2025 model scoring.
    for tag in selection['confirm_single_candidates']:
        cfg=selection['all_configs'][tag]
        cmd=[sys.executable,'-m','algae.experiment','run','--year','2025','--stage','holdout','--save',
             '--model',cfg['model'],'--objective',cfg['objective'],'--features',cfg['features'],
             '--season',cfg['season'],'--since',str(cfg['since'])]
        subprocess.run(cmd,cwd=ROOT,check=True)
    data,_=get_data()
    valid=data[data.split.eq('valid')].copy()
    test=data[data.split.eq('test')].copy()
    assert test.target.isna().all()
    z=predict_saved(selection['weights'],valid)
    report={
        'selection_sha256':selection_hash,'selected_weights':selection['weights'],
        'cv_mean_season_rmse':selection['cv_mean_season_rmse'],
        'selection_metric':selection['selection_metric'],
        'holdout_used_to_change_selection':False,'test_labels_scored':False,
        'training_policy':'Official train only; target_date < 2025-01-01; no valid-year refit.',
        'tabpfn_configuration':'Public TabPFN-v2, CPU, one estimator (not latest TabPFN).',
        'known_limitations':['Official combined metric and permission for valid-year refit unspecified.',
                             'Daily inputs lagged one day; algae assumed available on measurement date.',
                             'Rainfall blanks remain unknown; observed sums/fractions used.',
                             'Two conservative dam signals per station; detailed river network not modelled.']}
    score_rows=[]
    for scope,mask in [('all',np.ones(len(valid),dtype=bool)),('May_Oct',valid.base_date.dt.month.between(5,10).to_numpy())]:
        y=valid.target.to_numpy()[mask]
        report[scope]={'selected':metrics(y,z[mask]),
                       'persistence':metrics(y,valid.cells_log_lag0.to_numpy()[mask])}
        for model,item in report[scope].items():
            score_rows.append({'scope':scope,'model':model,**item})
        for tag in selection['confirm_single_candidates']:
            pred=pd.read_csv(model_folder(tag)/'predictions.csv')
            assert pred.station_id.tolist()==valid.station_id.tolist()
            assert pred.base_date.tolist()==valid.base_date.dt.strftime('%Y-%m-%d').tolist()
            item=metrics(y,pred.pred_log10.to_numpy()[mask])
            score_rows.append({'scope':scope,'model':tag,**item})
    valid_output=valid[['station_id','base_date','target_date','target','cells_log_lag0']].copy()
    valid_output['pred_log10']=z
    valid_output['pred_cells']=10**z-1
    valid_output.to_csv(outputs/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    report['station_season']={site:metrics(g.target,g.pred_log10) for site,g in
                             valid_output[valid_output.base_date.dt.month.between(5,10)].groupby('station_id')}
    pd.DataFrame(score_rows).to_csv(outputs/'validation_scores.csv',index=False,encoding='utf-8-sig')
    test_z=predict_saved(selection['weights'],test)
    submission=load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')['sample_submission'].copy()
    for col in ['station_id','base_date','target_date']:
        actual=test[col].dt.strftime('%Y-%m-%d').tolist() if col!='station_id' else test[col].tolist()
        assert submission[col].tolist()==actual
    submission['pred_cells']=np.rint(np.maximum(10**test_z-1,0)).astype(np.int64)
    assert submission.pred_cells.ge(0).all() and len(submission)==124
    assert not submission.duplicated(['station_id','base_date','target_date']).any()
    submission.to_csv(outputs/'submission_draft.csv',index=False,encoding='utf-8')
    reread=pd.read_csv(outputs/'submission_draft.csv')
    assert list(reread)==['station_id','base_date','target_date','pred_cells']
    assert pd.api.types.is_integer_dtype(reread.pred_cells)
    np.testing.assert_array_equal(reread.pred_cells,submission.pred_cells)
    report['submission_rows']=len(submission)
    report['submission_sha256']=hashlib.sha256((outputs/'submission_draft.csv').read_bytes()).hexdigest()
    report['submission_checks']='Keys/order, row count, integer dtype, finite/nonnegative values, saved-model inference.'
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'algae').glob('*.py'))}
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==selection_hash
    (outputs/'run_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    chart(valid_output,outputs)
    print(json.dumps(report,indent=2),flush=True)


def chart(valid,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True)
    names={'HP':'Haepyeong','GJGR':'Gangjeong-Goryeong','CS':'Chilseo','MGMR':'Mulgeum-Maeri','DCHN':'Hoenam'}
    warm=valid[valid.base_date.dt.month.between(5,10)]
    for ax,(site,g) in zip(axes,warm.groupby('station_id')):
        g=g.sort_values('target_date')
        ax.plot(g.target_date,np.log10(g.target+1),'o-',label='Observed',color='#202020',markersize=3)
        ax.plot(g.target_date,g.pred_log10,'o-',label='Selected model',color='#087f8c',markersize=3)
        ax.plot(g.target_date,g.cells_log_lag0,'--',label='Persistence',color='#b7b7b7',alpha=.8)
        ax.set_title(names[site],loc='left',fontsize=11)
        ax.set_ylabel('log10(cells+1)')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=9)
    fig.suptitle('2025 holdout: forecasts selected using 2021-2024 only',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97])
    fig.savefig(output/'validation_2025.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
