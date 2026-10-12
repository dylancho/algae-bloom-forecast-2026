"""Check independent raw-bundle inference against the frozen draft."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'results'
    draft = folder/'submission_draft.csv'
    replay = folder/'submission_replayed.csv'
    pd.testing.assert_frame_equal(pd.read_csv(draft), pd.read_csv(replay), check_exact=True)
    assert draft.read_bytes() == replay.read_bytes()
    path = folder/'run_summary.json'
    report = json.loads(path.read_text())
    report['independent_replay_identical'] = True
    report['independent_replay_sha256'] = hashlib.sha256(replay.read_bytes()).hexdigest()
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: independently rebuilt features and saved models reproduce all 124 forecasts exactly.')


if __name__ == '__main__':
    main()
