"""Download public TabPFN-v2 weights; no competition data is transmitted."""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
destination=ROOT/'.cache/tabpfn'
destination.mkdir(parents=True,exist_ok=True)
files=['tabpfn-v2-regressor-v2_default.ckpt','LICENSE.txt']
manifest={}
for name in files:
    url='https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/'+name
    path=destination/name
    if not path.exists():
        with requests.get(url,stream=True,timeout=60) as response:
            response.raise_for_status()
            temp=path.with_suffix(path.suffix+'.partial')
            with temp.open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    f.write(chunk)
            temp.replace(path)
    manifest[name]={'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(name,path.stat().st_size,flush=True)
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
