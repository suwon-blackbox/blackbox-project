"""AI Hub 사고유형의 B 진행정보 글로 Stage 3 약한 정답을 만든다. 사람이 영상을 보지 않는다.

1인칭 영상에서는 내 차 = 사고객체 B. B 진행정보가 '좌회전'이면 충돌 직전 내 차는 대체로 좌회전 중이다.
0.1초 정답이 아니라 영상 단위 기대값이고 틀린 것이 섞여 있다 → 평가 방법은 docs/stage3_weak_eval.md

출력
  configs/stage3_weak_map.csv     B 진행정보 글 → 기대 라벨 매핑 (git 포함, 손으로 고쳐도 됨)
  data/splits/stage3_weak.csv     영상별 약한 정답
  data/splits/stage3_weak_check.csv  사람이 확인할 점검 목록 (test 분할에서, 담당자 배정)

실행:
  python scripts/make_weak_labels.py                 # 매핑표가 없으면 규칙으로 만들고, 있으면 그대로 사용
  python scripts/make_weak_labels.py --rebuild-map   # 규칙으로 매핑표를 다시 만들기 (손으로 고친 내용 덮어씀)
  python scripts/make_weak_labels.py --extract       # 점검 영상을 data/raw/stage3_weak_check/ 에 풀기
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_splits import extract_videos, hash_split  # noqa: E402
from src.common.inventory import load_videos  # noqa: E402
from src.common.paths import PROJECT_ROOT, load_paths  # noqa: E402

MAP_CSV = PROJECT_ROOT / "configs" / "stage3_weak_map.csv"
SEED = 42

# 방향이 정해지지 않거나 우리 클래스 밖인 경우 → 제외 (위에서부터 먼저 걸리는 규칙 적용)
EXCLUDE = [
    (r"또는|동일>|이륜차 (좌|우)측", "좌우 둘 다 가능"),
    (r"후진", "후진은 클래스 밖"),
    (r"진로\s*변경|차로\s*변경|추월|합류|회전교차로|교차로 내 회전|중앙선", "방향 미정 또는 짧은 조향"),
    (r"선행자동차\(1차사고차량\)|후행\s*추돌|낙하|보행자 전용", "움직임 미정"),
]
RULES = [
    # (패턴, expected_lat, expected_long, tier, 설명)
    (r"정차\s*중|주\(정\)차|주정차", "직진", "정속", "high", "멈춰 있음"),
    (r"정차 후 출발", "직진", "가속", "high", "멈췄다가 출발"),
    (r"유턴", "좌", None, "high", "유턴 = 좌회전 방향"),
    (r"좌회[전진]", "좌", None, "high", "좌회전"),
    (r"우회전", "우", None, "high", "우회전"),
    (r"직진|차도 주행|주행<", "직진", None, "medium", "직진 (작은 조향이 섞일 수 있음)"),
]


def classify(text: str) -> dict:
    # '직진(직진·우회전 노면표시차로)'의 '우회전'은 차로 표시이지 움직임이 아니다
    text = re.sub(r"\([^)]*노면표시[^)]*\)", "", text)
    for pat, why in EXCLUDE:
        if re.search(pat, text):
            return {"expected_lat": None, "expected_long": None, "tier": "exclude", "rule": why}
    has_left, has_right = re.search(r"좌회[전진]|유턴", text), re.search(r"우회전", text)
    if has_left and has_right:
        return {"expected_lat": None, "expected_long": None, "tier": "exclude", "rule": "좌우 둘 다 나옴"}
    for pat, lat, lon, tier, why in RULES:
        if re.search(pat, text):
            return {"expected_lat": lat, "expected_long": lon, "tier": tier, "rule": why}
    return {"expected_lat": None, "expected_long": None, "tier": "exclude", "rule": "규칙 없음"}


def build_map(videos: pd.DataFrame) -> pd.DataFrame:
    counts = videos["b_progress_name"].value_counts()
    rows = [{"b_progress_name": name, "n_videos": n, **classify(name)} for name, n in counts.items()]
    return pd.DataFrame(rows)


def pick_check_list(weak: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """test 분할에서 기대 라벨별로 고르게 뽑는다."""
    test = weak[weak["split_ours"] == "test"].copy()
    test["expect"] = test["expected_lat"] + "·" + test["expected_long"].fillna("-")
    parts = []
    for tier, n in [("high", cfg["check_n_high"]), ("medium", cfg["check_n_medium"])]:
        pool = test[test["tier"] == tier]
        groups = pool.groupby("expect")
        per = max(1, n // groups.ngroups)
        picked = groups.apply(lambda g: g.sample(min(len(g), per), random_state=SEED), include_groups=False)
        picked = picked.reset_index(level=0)
        if len(picked) < n:  # 적은 그룹이 모자란 만큼 나머지에서 채움
            rest = pool.drop(picked.index, errors="ignore")
            picked = pd.concat([picked, rest.sample(n - len(picked), random_state=SEED)])
        parts.append(picked.head(n))
    check = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    check["checker"] = [cfg["checkers"][i % len(cfg["checkers"])] for i in range(len(check))]
    check["check_result"] = ""   # 사람이 채움: yes(기대대로 움직임) / no / unclear
    check["memo"] = ""
    return check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-map", action="store_true")
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "stage3_weak.yaml").read_text(encoding="utf-8"))
    paths = load_paths()
    videos = load_videos()
    videos = videos[(videos["filming_way"] == "bb") & (videos["video_point_of_view"] == 1) & videos["type_known"]].copy()
    videos["split_ours"] = videos["video_name"].map(hash_split)

    if args.rebuild_map or not MAP_CSV.exists():
        build_map(videos).to_csv(MAP_CSV, index=False, encoding="utf-8-sig")
        print(f"매핑표 생성 → {MAP_CSV}")
    mapping = pd.read_csv(MAP_CSV, encoding="utf-8-sig")

    weak = videos.merge(mapping.drop(columns=["n_videos"]), on="b_progress_name", how="left")
    weak = weak[weak["tier"].isin(["high", "medium"])]
    cols = ["video_name", "split_ours", "accident_object_name", "traffic_accident_type", "b_progress_name",
            "expected_lat", "expected_long", "tier", "rule", "source_zip", "video_file"]
    out_dir = paths["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    weak[cols].to_csv(out_dir / "stage3_weak.csv", index=False, encoding="utf-8-sig")

    check = pick_check_list(weak, cfg)
    check_cols = ["video_name", "b_progress_name", "expected_lat", "expected_long", "tier",
                  "source_zip", "video_file", "checker", "check_result", "memo"]
    check_path = out_dir / "stage3_weak_check.csv"
    if check_path.exists():
        print(f"[유지] 점검 목록이 이미 있어 덮어쓰지 않음 (사람이 채운 결과 보호): {check_path}")
    else:
        check[check_cols].to_csv(check_path, index=False, encoding="utf-8-sig")

    # 요약 (집계만)
    print(f"매핑: B 진행정보 {len(mapping)}종 → {mapping['tier'].value_counts().to_dict()}")
    print(f"약한 정답 {len(weak)}건 → {out_dir / 'stage3_weak.csv'}  (1인칭 {len(videos)}건 중)")
    t = weak[weak["split_ours"] == "test"]
    print(f"  test 분할 {len(t)}건: 신뢰도 {t['tier'].value_counts().to_dict()}")
    print(f"  test 기대 좌우: {t['expected_lat'].value_counts().to_dict()}")
    print(f"  test 기대 종방향: {t['expected_long'].value_counts().to_dict()}")
    print(f"점검 목록 {len(check)}건 → {check_path}  담당 {check['checker'].value_counts().to_dict()}")

    if args.extract:
        extract_videos(pd.read_csv(check_path, encoding="utf-8-sig"), paths["aihub_root"],
                       paths["raw_dir"] / "stage3_weak_check")
        print(f"점검 영상 → {paths['raw_dir'] / 'stage3_weak_check'}")


if __name__ == "__main__":
    main()
