"""Convert NIER national water-quality network xlsx snapshots to one tidy CSV.

Source: 물환경정보시스템 > 수질측정망 > 파일셋 다운로드 (water.nier.go.kr/web/board/29/?pMENU_NO=572),
saved under data_external/nier_wq/*.xlsx by hand-run curl (see manifest.json).
Only river/lake sheets of the Nakdong and Geum basins are kept; values are parsed as float,
'정량한계미만' (below quantification limit) becomes NaN with a separate flag column dropped.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'data_external/nier_wq'
OUT = FOLDER/'wq_nakdong_geum.csv'

# substring of the xlsx header -> short column name
WANT = {
    '채수 일자': 'date', '측정소 명': 'site', '분류번호': 'code', '수계 명': 'basin', '중권역 명': 'subbasin',
    '수온': 'wtemp', '용존산소': 'do', '수소이온': 'ph', '유량': 'flow', '전기전도도': 'ec',
    '생물화학적산소요구량': 'bod', '화학적산소요구량': 'cod', '총유기탄소': 'toc', '부유물질': 'ss',
    '총질소': 'tn', '총인': 'tp', '용존총질소': 'dtn', '용존총인': 'dtp',
    '암모니아성 질소': 'nh3n', '질산성': 'no3n', '인산염': 'po4p', '클로로필': 'chla', '투명도': 'secchi',
}


def map_header(h):
    h = str(h or '').strip()
    # exact-ish ordering: longer keys first so '용존총질소' wins over '총질소'
    for k in sorted(WANT, key=len, reverse=True):
        if k in h:
            return WANT[k]
    return None


def parse_sheet(ws, kind):
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    cols = [map_header(h) for h in header]
    keep = [(i, c) for i, c in enumerate(cols) if c]
    out = []
    for r in rows:
        rec = {c: r[i] if i < len(r) else None for i, c in keep}
        if rec.get('basin') not in ('낙동강', '금강'):
            continue
        rec['kind'] = kind
        out.append(rec)
    return out


def main():
    recs = []
    for f in sorted(FOLDER.glob('*.xlsx')):
        wb = openpyxl.load_workbook(f, read_only=True)
        for ws in wb.worksheets:
            if ws.title not in ('하천', '호소'):
                continue
            part = parse_sheet(ws, ws.title)
            for p in part:
                p['file'] = f.name
            recs += part
            print(f.name, ws.title, len(part), flush=True)
    df = pd.DataFrame(recs)
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip().str.replace('.', '-', regex=False), errors='coerce')
    num = [c for c in df.columns if c not in ('date', 'site', 'code', 'basin', 'subbasin', 'kind', 'file')]
    for c in num:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip().replace({'': None, 'None': None}), errors='coerce')
    # 대청댐5 was renamed 대청댐5(대청호) from 2024-12; same station
    df['site'] = df['site'].astype(str).str.strip().replace({'대청댐5(대청호)': '대청댐5'})
    df = df.dropna(subset=['date']).sort_values(['site', 'date'])
    # overlapping yearly files: keep the latest file's version of the same (site, date, kind) row set
    df = df.drop_duplicates(subset=['site', 'kind', 'date'] + [c for c in ['tn', 'tp', 'chla', 'wtemp'] if c in df], keep='last')
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print('rows', len(df), 'date range', df.date.min(), df.date.max())
    print(df.groupby('file').date.agg(['min', 'max', 'count']))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
