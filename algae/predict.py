"""Rebuild causal features from a bundle and replay saved models without fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from algae.features import build_features,load_bundle
from algae.finalize import predict_saved
from algae.experiment import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=ROOT/'02_dataset_bundle_algae2026.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'results/submission_replayed.csv')
    args=parser.parse_args()
    tables=load_bundle(args.bundle)
    template=tables['sample_submission'].copy()
    features,provenance=build_features(tables,template)
    assert (provenance.weather_latest_source<provenance.base_date).all()
    assert (provenance.dam_latest_source<provenance.base_date).all()
    assert (provenance.algae_latest_source<=provenance.base_date).all()
    selection=json.loads((ROOT/'artifacts/selection.json').read_text())
    z=predict_saved(selection['weights'],features)
    template['pred_cells']=np.rint(np.maximum(10**z-1,0)).astype(np.int64)
    assert np.isfinite(template.pred_cells).all()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    template.to_csv(args.output,index=False,encoding='utf-8')
    print('Replayed',len(template),'forecasts:',args.output)


if __name__=='__main__':
    main()
