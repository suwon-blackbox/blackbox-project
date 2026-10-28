"""AI Hub 데이터설명서 PDF의 [별첨 ②] 사고유형 표를 CSV로 뽑는다.

라벨 JSON의 traffic_accident_type(0~433) 하나가 아래 조합 하나에 대응한다.
  사고객체 / 사고장소 / 사고장소특징 / A진행방향 / B진행방향 / 과실비율 A / 과실비율 B

출력: configs/accident_types.csv (git에 포함 — 코드 의미표라 데이터가 아님)

실행:
  pip install pdfplumber
  python scripts/parse_accident_types.py "1-56 교통사고 영상 데이터_데이터설명서(수정)_250328.pdf"
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "configs" / "accident_types.csv"
N_TYPES = 434  # 0 ~ 433
OBJECTS = ["차대차", "차대보행자", "차대자전거", "차대이륜차"]
INT_RE = re.compile(r"^\d+$")

# 페이지 경계에 걸려 표 추출이 안 되는 행. PDF 본문 텍스트(p10, p11)를 보고 직접 옮겼다.
MANUAL_ROWS = {
    94: ("차대차", "사거리교차로(신호등있음)", "유턴 구역", "우회전(좌측도로)", "신호에 따른 유턴", 80, 20),
    140: ("차대보행자", "횡단보도(신호등있음)", "자동차 교차로 통과 후",
          "[적색신호] 횡단 시작, [적색신호] 충격", "[황색 우회전 신호] 우회전", 50, 50),
}


def normalize_place(s: str) -> str:
    """'사거리 교차로 (신 호등 있음)' 같은 표기 차이를 없애려고 공백을 모두 지운다."""
    return re.sub(r"\s+", "", s)


def clean(cell) -> str:
    """셀 안 줄바꿈과 PDF 글자 사이 공백을 정리한다."""
    if cell is None:
        return ""
    s = re.sub(r"\s+", " ", str(cell)).strip()
    # '차대 이륜차' 처럼 객체명이 쪼개진 경우
    return s.replace("차대 이륜차", "차대이륜차")


def parse_row(cells: list[str]) -> dict | None:
    """끝이 [과실A, 과실B, 유형코드] 숫자 3개인 행만 사고유형 행으로 본다."""
    vals = [c for c in cells if c]
    if len(vals) < 7 or not all(INT_RE.match(v) for v in vals[-3:]):
        return None
    rate_a, rate_b, code = (int(v) for v in vals[-3:])
    if rate_a + rate_b != 100 or not 0 <= code < N_TYPES:
        return None
    texts = vals[:-3]
    obj = next((t for t in texts if t in OBJECTS), "")
    texts = [t for t in texts if t not in OBJECTS]
    if len(texts) < 4:
        return None
    place, feature, a_info, b_info = texts[-4:]
    return {
        "traffic_accident_type": code,
        "accident_object_name": obj,
        "place_name": place,
        "place_feature_name": feature,
        "a_progress_name": a_info,
        "b_progress_name": b_info,
        "rate_a": rate_a,
        "rate_b": rate_b,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    args = ap.parse_args()

    rows: dict[int, dict] = {}
    current_obj = ""
    with pdfplumber.open(args.pdf) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:
                    row = parse_row([clean(c) for c in raw])
                    if row is None:
                        continue
                    # 사고객체는 그룹 첫 행에만 적혀 있으므로 이어받는다
                    if row["accident_object_name"]:
                        current_obj = row["accident_object_name"]
                    else:
                        row["accident_object_name"] = current_obj
                    rows.setdefault(row["traffic_accident_type"], row)

    for code, (obj, place, feature, a_info, b_info, rate_a, rate_b) in MANUAL_ROWS.items():
        rows.setdefault(code, {
            "traffic_accident_type": code, "accident_object_name": obj,
            "place_name": place, "place_feature_name": feature,
            "a_progress_name": a_info, "b_progress_name": b_info,
            "rate_a": rate_a, "rate_b": rate_b,
        })

    # 고속도로 그룹(187~198)은 표에 사고객체 칸이 비어 있어 앞 그룹(차대보행자)을 잘못 이어받는다.
    # 인벤토리(zip 분류)로 확인하면 전부 차대차다.
    for row in rows.values():
        if normalize_place(row["place_name"]).startswith("고속도로"):
            row["accident_object_name"] = "차대차"

    # 장소명 정리. 셀이 페이지에서 잘려 '사거리교차로(신호등' 처럼 끝난 경우는
    # 바로 앞 코드의 장소명이 그 글자로 시작하면 이어받는다 (표가 장소별로 연속이라서).
    prev = ""
    for code in sorted(rows):
        place = normalize_place(rows[code]["place_name"])
        if not place.endswith(")") and prev.startswith(place):
            place = prev
        rows[code]["place_name"] = place
        prev = place

    missing = [c for c in range(N_TYPES) if c not in rows]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
        w.writeheader()
        w.writerows(rows[c] for c in sorted(rows))

    print(f"사고유형 {len(rows)}/{N_TYPES}개 → {OUT}")
    if missing:
        print(f"표에서 못 읽은 코드 {len(missing)}개: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
