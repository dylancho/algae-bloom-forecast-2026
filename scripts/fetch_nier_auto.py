"""Snapshot NIER automatic water-quality network (자동측정망) daily data near the 5 stations.

Endpoint is the public 물환경정보시스템 '확정 자료 조회' grid (querySn 63 = daily); the latest
~2 months come from the '미확정 자료 조회' grid instead. Raw JSON per station-year is saved
under data_external/nier_auto so feature building never calls the network again.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data_external/nier_auto'
B = 'https://water.nier.go.kr'
STATIONS = {'S02012': '해평', 'S02015': '칠곡', 'S02019': '구미', 'S02004': '강창', 'S02002': '고령',
            'S02005': '적포', 'S02007': '칠서', 'S02008': '창암', 'S02017': '상동', 'S02056': '양산',
            'S03003': '대청호', 'S03007': '장계', 'S03010': '옥천천', 'S03011': '이원', 'S03009': '현도'}


def session(page):
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'})
    s.get(B+page, timeout=60)
    return s


def query(s, kind, code, start, end):
    menu, url = {'confirm': ('576', '/web/autoMeasure/confirm/toastList'),
                 'recent': ('574', '/web/autoMeasure/noConfirm/toastList')}[kind]
    # daily grid: confirmed querySn 63, unconfirmed querySn 60 (Chl-a is not released unconfirmed)
    data = [('pMENU_NO', menu), ('page', ''), ('ATTR_2', start), ('ATTR_3', end),
            ('querySn', '63' if kind == 'confirm' else '60'),
            ('queryTp', 'autoMe'), ('confirmNum', '2' if kind == 'confirm' else '1'), ('station_code', code)]
    for attempt in range(4):
        try:
            r = s.post(B+url, data=data, timeout=120)
            return r.json()['list']
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5*(attempt+1))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    s = session('/web/autoMeasure/confirm?pMENU_NO=576')
    today = date.today()
    log = {'fetched_at': datetime.now().isoformat(timespec='seconds'), 'stations': STATIONS, 'files': {}}
    for code, name in STATIONS.items():
        for year in range(2012, today.year+1):
            f = OUT/f'{code}_{year}.json'
            if not f.exists() or year == today.year:
                rows = query(s, 'confirm', code, f'{year}0101', min(f'{year}1231', today.strftime('%Y%m%d')))
                f.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
                time.sleep(.3)
            rows = json.loads(f.read_text(encoding='utf-8'))
            log['files'][f.name] = sum(any(v not in (None, '') for k, v in r.items() if k.startswith('IEM_')) for r in rows)
        print(code, name, {y: log['files'][f'{code}_{y}.json'] for y in range(2012, today.year+1)}, flush=True)
    # unconfirmed recent window (last ~2 months) for the current year
    s2 = session('/web/autoMeasure/noConfirm?pMENU_NO=574')
    for code in STATIONS:
        rows = query(s2, 'recent', code, f'{today.year}0401', today.strftime('%Y%m%d'))
        (OUT/f'{code}_recent.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        log['files'][f'{code}_recent.json'] = len(rows)
    (OUT/'manifest.json').write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
