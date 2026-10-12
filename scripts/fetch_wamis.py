"""Snapshot WAMIS daily river water level and flow for gauges near the 5 stations.

WAMIS open API needs no key. Raw responses are saved as-is under
data_external/wamis so feature building never calls the network again
(2026 flow is provisional and can be revised later; the snapshot is what we use).
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data_external/wamis'
BASE = 'http://www.wamis.go.kr:8080/wamis/openapi/wkw'

# station -> gauges (upstream/near first). Names from wl_dubwlobs.
GAUGES = {
    '2011640': 'HP 구미시(구미대교)',
    '2011648': 'HP 칠곡군(칠곡보상)',
    '2011660': 'GJGR 성주군(성주대교)',
    '2011696': 'GJGR 대구시(강정고령보하)',
    '2014640': 'GJGR 고령군(고령교)',
    '2017620': 'CS 합천군(적포교)',
    '2020647': 'CS 창녕군(창녕함안보상)',
    '2022610': 'MGMR 밀양시(삼랑진교)',
    '3004650': 'DCHN 영동군(호탄리)',
    '3006680': 'DCHN 옥천군(이원대교)',
    '3007670': 'DCHN 옥천군(산계리)',
    '3008630': 'DCHN 옥천군(옥각교)',
}
YEARS = range(2012, 2027)


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as exc:  # network hiccups
            if attempt == 3:
                raise
            time.sleep(3*(attempt+1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = {'fetched_at': datetime.now().isoformat(timespec='seconds'), 'gauges': GAUGES, 'files': {}}
    for code in GAUGES:
        for year in YEARS:
            for kind, url in [('wl', f'{BASE}/wl_dtdata?obscd={code}&startdt={year}0101&enddt={min(f'{year}1231', datetime.now().strftime('%Y%m%d'))}&output=json'),
                              ('flw', f'{BASE}/flw_dtdata?obscd={code}&year={year}&output=json')]:
                f = OUT/f'{code}_{kind}_{year}.json'
                if not f.exists():
                    data = get(url)
                    f.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
                    time.sleep(.3)
                n = json.loads(f.read_text(encoding='utf-8')).get('count', 0)
                log['files'][f.name] = n
        print(code, GAUGES[code], {k: sum(v for f, v in log['files'].items() if f.startswith(f'{code}_{k}'))
                                   for k in ['wl', 'flw']}, flush=True)
    (OUT/'manifest.json').write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
