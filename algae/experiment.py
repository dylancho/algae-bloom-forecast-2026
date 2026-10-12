from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('SKB_DATA_DIRECTORY', str(ROOT/'.cache/skrub'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(ROOT/'.cache/tabpfn'))
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import joblib
import numpy as np
import pandas as pd
from algae.features import build_dataset, load_bundle, build_features


def metrics(y, z):
    z = np.maximum(np.asarray(z, dtype=float), 0)
    y = np.asarray(y, dtype=float)
    raw = np.power(10., z)-1
    error = z-np.log10(y+1)
    denominator = np.sum((y-y.mean())**2)
    return {'n': len(y), 'rmse_log10': float(np.sqrt(np.mean(error**2))),
            'mae_log10': float(np.abs(error).mean()),
            'r2_nse': float(1-np.sum((y-raw)**2)/denominator) if denominator else None}


def get_data(feature_set='full'):
    folder = ROOT/'artifacts/features'
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    names = manifest['features']
    if feature_set != 'full':
        allowed = {'algae':['algae'], 'water':['algae','water'],
                   'weather':['algae','water','weather']}[feature_set]
        names = [c for c in names if any(c in manifest['feature_groups'][k] for k in allowed)]
    data = pd.read_pickle(folder/'dataset.pkl')
    for col in ['base_date', 'target_date']:
        data[col] = pd.to_datetime(data[col])
    return data, names


def folds(data, year, season='all', since=2012):
    start = pd.Timestamp(year=year, month=1, day=1)
    train = data[data.split.eq('train') & data.base_date.lt(start) & data.target_date.lt(start)
                 & data.base_date.dt.year.ge(since)].copy()
    if season == 'warm':
        train = train[train.base_date.dt.month.between(4,11)]
    test = data[data.base_date.dt.year.eq(year) & data.split.isin(['train','valid'])].copy()
    assert len(train) and len(test)
    assert train.target_date.max() < start
    assert train.target.notna().all() and test.target.notna().all()
    assert not train.index.intersection(test.index).size
    return train, test


def make_model(name):
    if name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=600, depth=5, learning_rate=.04,
                                 l2_leaf_reg=8, loss_function='RMSE', has_time=True,
                                 random_seed=42, thread_count=4, verbose=False,
                                 allow_writing_files=False, cat_features=['station'])
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=500, num_leaves=15, max_depth=5,
                             min_child_samples=30, learning_rate=.035,
                             reg_lambda=10, reg_alpha=.1, colsample_bytree=.9,
                             random_state=42, n_jobs=4, verbosity=-1,
                             deterministic=True, force_col_wise=True)
    if name == 'tabpfn':
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        torch.set_num_threads(4)
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device='cpu', n_estimators=1, random_state=42,
            model_path=str(ROOT/'.cache/tabpfn/tabpfn-v2-regressor-v2_default.ckpt'),
            ignore_pretraining_limits=True, categorical_features_indices=[0],
            fit_mode='fit_preprocessors', n_preprocessing_jobs=1)
    raise ValueError(name)


def design(frame, names, model):
    x = frame[names].copy()
    x['station'] = x.station.astype(int)
    if model == 'lightgbm':
        x['station'] = pd.Categorical(x.station, categories=list(range(5)))
    return x


def fit_predict(model_name, objective, train, test, names):
    train = train.sort_values(['base_date','station_id'])
    y = np.log10(train.target.to_numpy(float)+1)
    if objective == 'delta':
        y -= train.cells_log_lag0.to_numpy(float)
    estimator = make_model(model_name)
    x_train, x_test = design(train,names,model_name), design(test,names,model_name)
    estimator.fit(x_train, y)
    z = np.asarray(estimator.predict(x_test),dtype=float)
    if objective == 'delta':
        z += test.cells_log_lag0.to_numpy(float)
    z = np.maximum(z,0)
    assert np.isfinite(z).all()
    return estimator, z


def run_job(args):
    tag = f'{args.model}_{args.objective}_{args.features}_{args.season}_{args.since}'
    output = ROOT/'artifacts'/args.stage/f'{args.year}_{tag}'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.json').exists() and not args.force:
        print('CACHED', output.name, flush=True)
        return
    config = vars(args).copy()
    config['tag'] = tag
    config['selection_note'] = 'Historical CV only; 2025 reserved for frozen candidate confirmation.'
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    data,names = get_data(args.features)
    train,test = folds(data,args.year,args.season,args.since)
    print('START',output.name,'train',len(train),'eval',len(test),'features',len(names),flush=True)
    started = time.time()
    try:
        estimator,z = fit_predict(args.model,args.objective,train,test,names)
        predictions = test[['station_id','base_date','target_date','target','cells_log_lag0']].copy()
        predictions['pred_log10'] = z
        predictions['pred_cells'] = np.power(10.,z)-1
        predictions.to_csv(output/'predictions.csv',index=False)
        warm = test.base_date.dt.month.between(5,10).to_numpy()
        report = {'tag':tag,'year':args.year,'n_train':len(train),'n_features':len(names),
                  'elapsed_seconds':time.time()-started,
                  'all':metrics(test.target,z),'season':metrics(test.target.to_numpy()[warm],z[warm]),
                  'station_season':{site:metrics(g.target,g.pred_log10) for site,g in
                                     predictions[predictions.base_date.dt.month.between(5,10)].groupby('station_id')}}
        if args.save:
            payload = {'model':estimator,'model_name':args.model,'objective':args.objective,
                       'features':names,'config':config,'train_target_max':str(train.target_date.max())}
            joblib.dump(payload,output/'model.joblib')
            restored = joblib.load(output/'model.joblib')
            replay = restored['model'].predict(design(test,names,args.model))
            if args.objective=='delta':
                replay += test.cells_log_lag0.to_numpy(float)
            np.testing.assert_allclose(np.maximum(replay,0),z,rtol=1e-7,atol=1e-7)
            report['serialization_replay_verified'] = True
        (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('DONE',output.name,json.dumps(report['season']),round(report['elapsed_seconds'],1),'sec',flush=True)
    except Exception as exc:
        (output/'error.txt').write_text(repr(exc),encoding='utf-8')
        raise


def summarize(stage='cv'):
    rows=[]
    for file in (ROOT/'artifacts'/stage).glob('*/metrics.json'):
        m=json.loads(file.read_text())
        rows.append({'tag':m['tag'],'year':m['year'],**m['season'],
                     'all_rmse_log10':m['all']['rmse_log10'],'seconds':m['elapsed_seconds']})
    frame=pd.DataFrame(rows)
    if len(frame):
        frame.sort_values(['tag','year']).to_csv(ROOT/'artifacts'/f'{stage}_scores.csv',index=False)
        print(frame.groupby('tag').agg(folds=('year','nunique'),mean_rmse=('rmse_log10','mean'),
                                      mean_mae=('mae_log10','mean'),mean_r2=('r2_nse','mean'),
                                      seconds=('seconds','sum')).sort_values('mean_rmse').to_string())
    return frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['build','run','summarize'])
    parser.add_argument('--model',choices=['catboost','lightgbm','tabpfn'],default='catboost')
    parser.add_argument('--objective',choices=['direct','delta'],default='direct')
    parser.add_argument('--year',type=int,default=2024)
    parser.add_argument('--features',choices=['algae','water','weather','full'],default='full')
    parser.add_argument('--season',choices=['all','warm'],default='all')
    parser.add_argument('--since',type=int,default=2012)
    parser.add_argument('--stage',default='cv')
    parser.add_argument('--save',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    if args.command=='build':
        m=build_dataset(ROOT/'02_dataset_bundle_algae2026.zip',ROOT/'artifacts/features')
        print('Built',m['rows'],'rows,',m['n_features'],'features')
        versions={p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','catboost','lightgbm','tabpfn','torch']}
        (ROOT/'artifacts/environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    elif args.command=='run':
        run_job(args)
    else:
        summarize(args.stage)


if __name__=='__main__':
    main()
