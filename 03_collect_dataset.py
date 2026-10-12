#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2026 DATA·AI 분석 경진대회 — 녹조 예측 문제 데이터 번들 수집 스크립트

대상 지점 5개소(해평·강정고령·칠서·물금매리·대청 회남)에 대한
조류경보제 / 다목적댐 / ASOS 자료를 CSV로 내려받는다.

사용 전
  1) 공공데이터포털 인증키(Decoding)를 환경변수로 설정
       PowerShell : $env:DATA_GO_KR_KEY = "발급받은 Decoding 키"
       bash       : export DATA_GO_KR_KEY="발급받은 Decoding 키"
  2) pip install requests
  3) 아래 3개 데이터에 활용신청이 되어 있어야 한다(자동승인).
       15126738 조류경보제 조회서비스 / 15099049 다목적댐 운영 정보 / 15059093 ASOS

사용법
  python collect_dataset.py probe                    # 응답 원문 1건 확인
  python collect_dataset.py algae  --from 2012 --to 2026
  python collect_dataset.py asos   --from 20120101 --to 20260912
  python collect_dataset.py dam    --from 2026-01-01 --to 2026-09-12
  python collect_dataset.py stations

'26. 9. 12. 실응답으로 확인한 사항
  - 세포수 단위는 cells/mL. 강정·고령 2026.08.18 = 483,021 로 K-water 주간보고서와 일치.
    API 명세의 'Cells/100㎖' 표기는 오기.
  - swmnNm 은 하천·호소명(낙동강/대청호)이고, 지점명은 swmnDetailNm 에 있다.
    따라서 지점 식별은 swmnCode 로 한다.
  - chckDe 는 'YYYY.MM.DD' 형식이라 ISO 로 변환해 저장한다.
  - 다목적댐 응답의 수치에는 천 단위 콤마가 섞여 있어 제거 후 저장한다.
  - 다목적댐 API는 개발계정 100건/일 제약이라 장기 시계열 수집에는 부적합하다.
    최근 구간만 API로 받고, 과거 구간은 파일데이터·MyWater 로 보완할 것.
"""

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_KEY = os.environ.get("DATA_GO_KR_KEY", "")

OUT_DIR = Path(__file__).parent / "algae_contest_2026"

ALGAE_URL = "https://apis.data.go.kr/1480523/nieragainstalgae/algaePreMeasure"
DAM_URL = "https://apis.data.go.kr/B500001/dam/multipurPoseDam/multipurPoseDamlist"
ASOS_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# WAMIS 개방 API — 인증키 불필요. api.wamis.go.kr 이 아니라 www 호스트로 접속해야 한다.
WAMIS_URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkd/mn_dtdata"

# 대상 지점 상류의 낙동강·금강 수계 다목적댐 및 조정지. 댐코드는 WAMIS mn_dammain 확인.
WAMIS_DAMS = [
    ("2001110", "안동",       "낙동강"),
    ("2001611", "안동조정지",  "낙동강"),
    ("2002110", "임하",       "낙동강"),
    ("2002610", "임하조정지",  "낙동강"),
    ("2002111", "성덕",       "낙동강"),
    ("2004101", "영주",       "낙동강"),
    ("2008110", "군위",       "낙동강"),
    ("2010101", "김천부항",    "낙동강"),
    ("2015110", "합천",       "낙동강"),
    ("2018110", "남강",       "낙동강"),
    ("2018611", "합천조정지",  "낙동강"),
    ("2021110", "밀양",       "낙동강"),
    ("3001110", "용담",       "금강"),
    ("3008110", "대청",       "금강"),
    ("3008611", "대청조정지",  "금강"),
]

# 대상 지점. swmn_code·asos 지점번호 모두 '26. 9. 12. 실응답으로 확인.
STATIONS = [
    {"swmn_code": "2011G26", "station_id": "HP",   "station_name": "해평",
     "swmn_nm": "낙동강", "suge": "낙동강", "river_lake": "하천", "section": "상수원",
     "asos_stn_id": "279", "asos_stn_nm": "구미",   "asos_alt": "",
     "service_start": "2016", "note": "구 칠곡"},
    {"swmn_code": "2011G56", "station_id": "GJGR", "station_name": "강정·고령",
     "swmn_nm": "낙동강", "suge": "낙동강", "river_lake": "하천", "section": "상수원",
     "asos_stn_id": "143", "asos_stn_nm": "대구",   "asos_alt": "",
     "service_start": "2016", "note": ""},
    {"swmn_code": "2020G33", "station_id": "CS",   "station_name": "칠서",
     "swmn_nm": "낙동강", "suge": "낙동강", "river_lake": "하천", "section": "상수원",
     "asos_stn_id": "263", "asos_stn_nm": "의령군", "asos_alt": "155(창원)",
     "service_start": "2016", "note": "구 창녕·함안"},
    {"swmn_code": "2022G05", "station_id": "MGMR", "station_name": "물금·매리",
     "swmn_nm": "낙동강", "suge": "낙동강", "river_lake": "하천", "section": "상수원",
     "asos_stn_id": "159", "asos_stn_nm": "부산",   "asos_alt": "",
     "service_start": "2020", "note": "'20년 조류경보제 편입"},
    {"swmn_code": "3008G30", "station_id": "DCHN", "station_name": "회남",
     "swmn_nm": "대청호", "suge": "금강", "river_lake": "호소", "section": "상수원",
     "asos_stn_id": "133", "asos_stn_nm": "대전",   "asos_alt": "",
     "service_start": "1998", "note": "대청호 3지점 중 1"},
]

BY_CODE = {s["swmn_code"]: s for s in STATIONS}

# 조류경보제 응답 21개 항목 → CSV 컬럼
ALGAE_FIELDS = [
    ("rowno",              "rowno"),
    ("swmnCode",           "swmn_code"),
    ("riverLkmhSe",        "river_lkmh_se"),
    ("swmnNm",             "swmn_nm"),
    ("swmnDetailNm",       "swmn_detail_nm"),
    ("detailAdres",        "detail_adres"),
    ("chckDe",             "chck_de"),
    ("iemWtrtp",           "iem_wtrtp"),
    ("iemPh",              "iem_ph"),
    ("iemDoc",             "iem_doc"),
    ("iemTrp",             "iem_trp"),
    ("iemTur",             "iem_tur"),
    ("iemChla",            "iem_chla"),
    ("iemBgalageCellCo",   "iem_bgalage_cell_co"),
    ("iemBgalageMicrosts", "iem_bgalage_microsts"),
    ("iemBgalageAnba",     "iem_bgalage_anba"),
    ("iemBgalageOsrtria",  "iem_bgalage_osrtria"),
    ("iemBgalageApzo",     "iem_bgalage_apzo"),
    ("iemGeosm",           "iem_geosm"),
    ("iemMib2",            "iem_mib2"),
    ("iemMicrostlr",       "iem_microstlr"),
]

DAM_FIELDS = [
    "suge", "damnm",
    "zerosevenhourprcptqy", "prcptqy", "pyacurf", "vyacurf", "oyaacurf",
    "inflowqy", "totdcwtrqy", "totdcwtrqyjo",
    "nowlowlevel", "lastlowlevel", "nyearlowlevel",
    "nowrsvwtqy", "lastrsvwtqy", "nyearrsvwtqy",
    "rsvwtrt", "dvlpqyacmtlacmslt",
]

ASOS_FIELDS = [
    "stnId", "stnNm", "tm",
    "avgTa", "maxTa", "minTa",
    "sumRn", "hr1MaxRn",
    "avgWs", "maxWs", "maxInsWs",
    "avgRhm", "sumSsHr", "sumGsr",
    "avgTd", "avgPs", "avgTca", "avgTs",
]

GENUS_COLS = [
    ("iem_bgalage_microsts", "Microcystis"),
    ("iem_bgalage_anba",     "Anabaena"),
    ("iem_bgalage_osrtria",  "Oscillatoria"),
    ("iem_bgalage_apzo",     "Aphanizomenon"),
]

CAUTION_THRESHOLD = 10_000        # 경계 기준 cells/mL


# ─────────────────────────────────────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────────────────────────────────────

def require_key():
    if not SERVICE_KEY:
        sys.exit("환경변수 DATA_GO_KR_KEY 가 비어 있다. 공공데이터포털 Decoding 키를 설정할 것.")


def get(url, params, retries=3, timeout=40):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:          # noqa: BLE001 — 원인 구분 없이 재시도
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"요청 실패: {url} — {last}")


def parse_items(xml_text):
    root = ET.fromstring(xml_text)
    code = root.findtext(".//resultCode")
    if code not in (None, "00", "0", "NORMAL_SERVICE"):
        raise RuntimeError(f"API 오류 resultCode={code} resultMsg={root.findtext('.//resultMsg')}")
    return [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("item")]


def iso_date(s):
    """'2026.08.24' → '2026-08-24'. 이미 ISO면 그대로."""
    s = (s or "").strip()
    return s.replace(".", "-") if "." in s else s


def denum(s):
    """'1,014.1' → '1014.1'. 천 단위 콤마 제거."""
    return (s or "").replace(",", "").strip()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.name}  ({len(rows):,}행)")


# ─────────────────────────────────────────────────────────────────────────────

def cmd_probe(args):
    """응답 원문을 저장해 스키마·단위를 눈으로 확인한다."""
    require_key()
    r = get(ALGAE_URL, {
        "serviceKey": SERVICE_KEY, "pageNo": "1", "numOfRows": "10", "resultType": "xml",
        "ptNoList": "", "wmyrList": "2026", "wmodList": "08",
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_probe_algae.xml").write_text(r.text, encoding="utf-8")
    items = parse_items(r.text)
    print(f"원문 저장: {OUT_DIR / '_probe_algae.xml'}  ({len(items)}건)")
    for it in items[:8]:
        print(f"  {it.get('swmnCode','?'):8s} {it.get('swmnNm','?'):6s} "
              f"{it.get('swmnDetailNm','?'):10s} {it.get('chckDe','?'):12s} "
              f"세포수={it.get('iemBgalageCellCo',''):>10s}")


# ─────────────────────────────────────────────────────────────────────────────

def cmd_algae(args):
    """조류경보제 측정자료. 연도·월 다중 조회를 지원하므로 연 단위로 끊는다."""
    require_key()
    months = ",".join(f"{m:02d}" for m in range(1, 13))
    collected, seen = [], set()

    for year in range(int(args.year_from), int(args.year_to) + 1):
        page, got = 1, 0
        while True:
            items = parse_items(get(ALGAE_URL, {
                "serviceKey": SERVICE_KEY, "pageNo": str(page), "numOfRows": "1000",
                "resultType": "xml", "ptNoList": "",
                "wmyrList": str(year), "wmodList": months,
            }).text)
            if not items:
                break

            for it in items:
                meta = BY_CODE.get(it.get("swmnCode", ""))
                if meta is None:
                    continue
                row = {"station_id": meta["station_id"], "station_name": meta["station_name"]}
                for src, dst in ALGAE_FIELDS:
                    row[dst] = it.get(src, "")
                row["chck_de"] = iso_date(row["chck_de"])
                key = (row["station_id"], row["chck_de"])
                if key in seen:
                    continue
                seen.add(key)
                row["dominant_genus"] = dominant_genus(row)
                row["over_caution"] = over_caution(row)
                collected.append(row)
                got += 1

            if len(items) < 1000:
                break
            page += 1
            time.sleep(0.2)
        print(f"  {year}: {got}행")

    collected.sort(key=lambda r: (r["station_id"], r["chck_de"]))
    fields = (["station_id", "station_name"] + [d for _, d in ALGAE_FIELDS]
              + ["dominant_genus", "over_caution"])
    write_csv(OUT_DIR / "algae_measure.csv", collected, fields)

    print("\n[지점별 수록 현황]")
    for s in STATIONS:
        rows = [r for r in collected if r["station_id"] == s["station_id"]]
        if rows:
            print(f"  {s['station_id']:5s} {s['station_name']:10s} {len(rows):>5,}행  "
                  f"{rows[0]['chck_de']} ~ {rows[-1]['chck_de']}")
        else:
            print(f"  {s['station_id']:5s} {s['station_name']:10s}     0행")


def _num(raw):
    try:
        return float(denum(raw))
    except ValueError:
        return None


def dominant_genus(row):
    """4개 속별 세포수 중 최댓값을 우점종으로 본다. 전부 비었으면 공란."""
    best, best_val = "", -1.0
    for col, name in GENUS_COLS:
        val = _num(row.get(col))
        if val is not None and val > best_val:
            best, best_val = name, val
    return best if best_val > 0 else ""


def over_caution(row):
    """경계 기준(10,000 cells/mL) 초과 여부. 순위 미반영 참고지표."""
    val = _num(row.get("iem_bgalage_cell_co"))
    return "" if val is None else ("1" if val >= CAUTION_THRESHOLD else "0")


# ─────────────────────────────────────────────────────────────────────────────

def cmd_dam(args):
    """다목적댐 일자료 — WAMIS 개방 API(인증키 불필요, 트래픽 제약 없음).

    공공데이터포털 다목적댐 API(15099049)는 개발계정 100건/일 제약이라
    10년치 일자료 수집이 불가능하다. WAMIS는 같은 항목을 기간 조회로 제공하므로
    장기 시계열은 이쪽을 주 출처로 쓰고, 저수량·저수율·전년 비교값이 필요하면
    damapi 서브커맨드로 최근 구간을 보완한다.

    요청당 약 3년(1,096일)까지만 반환하므로 연 단위로 끊어 받는다.
    """
    start, end = date.fromisoformat(args.date_from), date.fromisoformat(args.date_to)
    rows = []
    for code, name, suge in WAMIS_DAMS:
        got = 0
        for year in range(start.year, end.year + 1):
            y0 = max(start, date(year, 1, 1)).strftime("%Y%m%d")
            y1 = min(end, date(year, 12, 31)).strftime("%Y%m%d")
            r = get(WAMIS_URL, {"damcd": code, "startdt": y0, "enddt": y1, "output": "json"})
            for it in (r.json().get("list") or []):
                d = it["obsymd"]
                rows.append({
                    "obs_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                    "dam_code": code, "dam_name": name, "suge": suge,
                    "rwl": it.get("rwl", ""),           # 저수위 EL.m
                    "iqty": it.get("iqty", ""),         # 유입량 ㎥/s
                    "tdqty": it.get("tdqty", ""),       # 총방류량 ㎥/s
                    "edqty": it.get("edqty", ""),       # 발전방류량
                    "spdqty": it.get("spdqty", ""),     # 여수로방류량
                    "otltdqty": it.get("otltdqty", ""), # 기타방류량
                    "itqty": it.get("itqty", ""),       # 취수량
                    "rf": it.get("rf", ""),             # 강우량 mm
                })
                got += 1
            time.sleep(0.2)
        print(f"  {name}({code}): {got:,}행")

    rows.sort(key=lambda r: (r["dam_code"], r["obs_date"]))
    write_csv(OUT_DIR / "dam_daily.csv", rows,
              ["obs_date", "dam_code", "dam_name", "suge",
               "rwl", "iqty", "tdqty", "edqty", "spdqty", "otltdqty", "itqty", "rf"])


def cmd_damapi(args):
    """공공데이터포털 다목적댐 운영 정보(15099049) — 보조.

    저수량·저수율과 전년·예년 비교값을 제공하지만 개발계정 100건/일 제약이 있다.
    이미 받은 날짜는 건너뛰므로, 한도에 걸리면 다음 날 같은 명령으로 이어받는다.
    """
    require_key()
    start, end = date.fromisoformat(args.date_from), date.fromisoformat(args.date_to)
    out = OUT_DIR / "dam_operation_api.csv"
    fields = ["obs_date"] + DAM_FIELDS

    rows, done = [], set()
    if out.exists():
        with out.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add(row["obs_date"])
        print(f"기수집 {len(done)}일분을 이어받는다.")

    budget, cur, fetched = args.limit, start, 0
    while cur <= end and budget > 0:
        iso = cur.isoformat()
        if iso in done:
            cur += timedelta(days=1)
            continue
        try:
            items = parse_items(get(DAM_URL, {
                "serviceKey": SERVICE_KEY, "pageNo": "1", "numOfRows": "50",
                "vdate": iso, "tdate": (cur - timedelta(days=1)).isoformat(),
                "ldate": cur.replace(year=cur.year - 1).isoformat(),
                "vtime": args.vtime, "_type": "xml",
            }).text)
        except Exception as exc:                # noqa: BLE001
            print(f"  {iso}: 중단 — {exc}")
            break

        for it in items:
            if args.suge and it.get("suge") not in args.suge:
                continue
            row = {"obs_date": iso}
            row.update({k: denum(it.get(k, "")) for k in DAM_FIELDS})
            rows.append(row)

        budget -= 1
        fetched += 1
        cur += timedelta(days=1)
        time.sleep(0.2)

    print(f"  이번 실행 {fetched}일분 수집")
    rows.sort(key=lambda r: (r["obs_date"], r.get("damnm", "")))
    write_csv(out, rows, fields)
    if cur <= end:
        print(f"\n남은 구간: {cur.isoformat()} ~ {end.isoformat()}")
        print("트래픽 한도(100건/일) 초과 시 다음 날 같은 명령으로 이어받을 것.")


# ─────────────────────────────────────────────────────────────────────────────

def cmd_asos(args):
    """ASOS 일자료. 대상 지점의 최근접 관측지점(대체 지점 포함)을 모두 받는다."""
    require_key()
    ids = []
    for s in STATIONS:
        ids.append(s["asos_stn_id"])
        if s["asos_alt"]:
            ids.append(s["asos_alt"].split("(")[0])
    ids = sorted(set(i for i in ids if i), key=int)
    print(f"대상 지점번호: {', '.join(ids)}")

    rows = []
    for stn in ids:
        for y0, y1 in year_chunks(args.date_from, args.date_to):
            page = 1
            while True:
                items = parse_items(get(ASOS_URL, {
                    "serviceKey": SERVICE_KEY, "pageNo": str(page), "numOfRows": "999",
                    "dataType": "XML", "dataCd": "ASOS", "dateCd": "DAY",
                    "startDt": y0, "endDt": y1, "stnIds": stn,
                }).text)
                if not items:
                    break
                rows.extend({k: it.get(k, "") for k in ASOS_FIELDS} for it in items)
                if len(items) < 999:
                    break
                page += 1
                time.sleep(0.2)
        print(f"  {stn}: 누적 {len(rows):,}행")

    rows.sort(key=lambda r: (r.get("stnId", ""), r.get("tm", "")))
    write_csv(OUT_DIR / "asos_daily.csv", rows, ASOS_FIELDS)


def year_chunks(start_yyyymmdd, end_yyyymmdd):
    s = date(int(start_yyyymmdd[:4]), int(start_yyyymmdd[4:6]), int(start_yyyymmdd[6:8]))
    e = date(int(end_yyyymmdd[:4]), int(end_yyyymmdd[4:6]), int(end_yyyymmdd[6:8]))
    # ASOS는 전일 자료까지만 제공한다("전날 자료까지 제공됩니다" resultCode 99).
    yesterday = date.today() - timedelta(days=1)
    if e > yesterday:
        print(f"  종료일을 전일({yesterday.isoformat()})로 조정")
        e = yesterday
    for year in range(s.year, e.year + 1):
        yield (max(s, date(year, 1, 1)).strftime("%Y%m%d"),
               min(e, date(year, 12, 31)).strftime("%Y%m%d"))


# ─────────────────────────────────────────────────────────────────────────────

def cmd_stations(args):
    """지점 메타 CSV."""
    fields = ["station_id", "station_name", "swmn_code", "swmn_nm", "river_lake",
              "section", "suge", "asos_stn_id", "asos_stn_nm", "asos_alt",
              "service_start", "note"]
    write_csv(OUT_DIR / "stations.csv", [{k: s.get(k, "") for k in fields} for s in STATIONS], fields)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="조류경보제 응답 원문 확인")

    a = sub.add_parser("algae", help="조류경보제 측정자료 수집")
    a.add_argument("--from", dest="year_from", default="2012")
    a.add_argument("--to", dest="year_to", default=str(date.today().year))

    d = sub.add_parser("dam", help="다목적댐 일자료 수집 (WAMIS, 권장)")
    d.add_argument("--from", dest="date_from", default="2012-01-01")
    d.add_argument("--to", dest="date_to", default=date.today().isoformat())

    da = sub.add_parser("damapi", help="다목적댐 운영 정보 수집 (공공데이터포털, 보조)")
    da.add_argument("--from", dest="date_from", default="2026-01-01")
    da.add_argument("--to", dest="date_to", default=date.today().isoformat())
    da.add_argument("--vtime", default="07")
    da.add_argument("--limit", type=int, default=95, help="1회 실행 호출 상한(기본 95)")
    da.add_argument("--suge", nargs="*", default=["낙동강", "금강"], help="수집할 수계. 비우면 전체")

    s = sub.add_parser("asos", help="ASOS 종관기상 일자료 수집")
    s.add_argument("--from", dest="date_from", default="20120101")
    s.add_argument("--to", dest="date_to", default=date.today().strftime("%Y%m%d"))

    sub.add_parser("stations", help="지점 메타 CSV 생성")

    args = p.parse_args()
    {"probe": cmd_probe, "algae": cmd_algae, "dam": cmd_dam, "damapi": cmd_damapi,
     "asos": cmd_asos, "stations": cmd_stations}[args.cmd](args)


if __name__ == "__main__":
    main()
