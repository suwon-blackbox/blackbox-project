"""우리 학습/평가 분할을 만들고, Stage 2 시점 라벨링 대상 300건을 뽑는다.

1) 영상 단위 분할 (video_split.csv)
   AI Hub의 train/val은 프레임 단위라 쓰지 않는다 (README 참고). 대신 영상 이름의 해시로
   20%를 평가용(test)으로 고정한다. 해시라서 인벤토리가 바뀌어도 기존 영상의 분할은 그대로다.

2) Stage 2 라벨링 대상 (stage2_labeling.csv)
   - 블랙박스(bb) + 1인칭 + 사고유형 코드가 설명서에 있는 영상만
   - 그룹별 할당량을 채우되, 그룹 안에서는 사고유형이 겹치지 않게 돌아가며 뽑는다
   - 차대차 중 '진로변경·차로변경·합류·진입'이 들어간 유형(끼어들기류)을 많이 뽑는다 → 고의사고 핵심
   - 50건은 2명이 중복 라벨링 (라벨러 간 기준 맞추기, 계획서 9장)

실행:
  python scripts/make_splits.py                       # CSV만 생성
  python scripts/make_splits.py --extract             # 라벨링 대상 영상을 data/raw/stage2_labeling/ 에 풀기
  python scripts/make_splits.py --annotators 정식 A B C D
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.inventory import load_videos  # noqa: E402
from src.common.paths import load_paths  # noqa: E402

TEST_PERCENT = 20
SEED = 42
CUT_IN_PATTERN = "진로변경|차로변경|합류|끼어들|진입"

# 그룹별 라벨링 할당량 (합계 300)
QUOTAS = {
    "차대차_끼어들기류": 120,
    "차대차_기타": 60,
    "차대이륜차": 50,
    "차대자전거": 35,
    "차대보행자": 35,
}
N_DOUBLE = 50


def hash_split(name: str) -> str:
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    return "test" if h % 100 < TEST_PERCENT else "train"


def labeling_group(row: pd.Series) -> str:
    obj = row["accident_object_name"]
    if obj != "차대차":
        return obj
    text = " ".join(str(row[c]) for c in ["place_feature_name", "a_progress_name", "b_progress_name"])
    return "차대차_끼어들기류" if pd.Series([text]).str.contains(CUT_IN_PATTERN).iloc[0] else "차대차_기타"


def sample_diverse(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """사고유형별로 한 건씩 돌아가며 뽑아 유형이 고르게 섞이게 한다."""
    shuffled = pool.sample(frac=1, random_state=seed)
    shuffled["_rank"] = shuffled.groupby("traffic_accident_type").cumcount()
    type_order = {t: i for i, t in enumerate(shuffled["traffic_accident_type"].drop_duplicates())}
    shuffled["_type_order"] = shuffled["traffic_accident_type"].map(type_order)
    picked = shuffled.sort_values(["_rank", "_type_order"]).head(n)
    return picked.drop(columns=["_rank", "_type_order"])


def assign_annotators(df: pd.DataFrame, names: list[str], seed: int) -> pd.DataFrame:
    """앞 N_DOUBLE건은 2명, 나머지는 1명. 1인당 건수가 같아지도록 순서대로 배정."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    first, second = [], []
    k = 0
    for i in range(len(df)):
        first.append(names[k % len(names)])
        k += 1
        if i < N_DOUBLE:
            second.append(names[k % len(names)])
            k += 1
        else:
            second.append("")
    df["annotator_1"] = first
    df["annotator_2"] = second
    return df


def extract_videos(df: pd.DataFrame, aihub_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for zip_name, group in tqdm(df.groupby("source_zip"), desc="압축 해제"):
        zpath = next(aihub_root.rglob(zip_name))
        with zipfile.ZipFile(zpath) as z:
            for member in group["video_file"]:
                target = out_dir / Path(member).name
                if not target.exists():
                    target.write_bytes(z.read(member))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--extract", action="store_true", help="라벨링 대상 영상을 data/raw/stage2_labeling/ 에 풀기")
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = load_videos()
    videos["split_ours"] = videos["video_name"].map(hash_split)
    videos[["video_name", "split_ours"]].to_csv(out_dir / "video_split.csv", index=False, encoding="utf-8-sig")

    pool = videos[
        (videos["filming_way"] == "bb")
        & (videos["video_point_of_view"] == 1)
        & videos["type_known"]
    ].copy()
    pool["label_group"] = pool.apply(labeling_group, axis=1)

    picked = []
    for i, (group, n) in enumerate(QUOTAS.items()):
        g = pool[pool["label_group"] == group]
        if len(g) < n:
            print(f"[경고] {group}: 후보 {len(g)}건 < 할당 {n}건", file=sys.stderr)
        picked.append(sample_diverse(g, n, SEED + i))
    labeling = assign_annotators(pd.concat(picked), args.annotators, SEED)

    cols = ["video_name", "split_ours", "label_group", "accident_object_name", "traffic_accident_type",
            "place_name", "place_feature_name", "a_progress_name", "b_progress_name",
            "accident_negligence_rateA", "accident_negligence_rateB",
            "source_zip", "video_file", "video_size_mb", "annotator_1", "annotator_2"]
    labeling[cols].to_csv(out_dir / "stage2_labeling.csv", index=False, encoding="utf-8-sig")

    # 요약 (집계만 출력)
    print(f"영상 분할 {len(videos)}건 → {out_dir / 'video_split.csv'}")
    print(f"  {videos['split_ours'].value_counts().to_dict()}")
    print(f"라벨링 대상 {len(labeling)}건 → {out_dir / 'stage2_labeling.csv'}  (후보 {len(pool)}건)")
    print(f"  그룹: {labeling['label_group'].value_counts().to_dict()}")
    print(f"  분할: {labeling['split_ours'].value_counts().to_dict()}")
    print(f"  사고유형 종류: {labeling['traffic_accident_type'].nunique()}개")
    workload = pd.concat([labeling["annotator_1"], labeling["annotator_2"]]).loc[lambda s: s != ""]
    print(f"  1인당 건수: {workload.value_counts().to_dict()} (중복 {N_DOUBLE}건 포함)")
    print(f"  필요한 원천 zip {labeling['source_zip'].nunique()}개, 영상 합계 {labeling['video_size_mb'].sum():.0f} MB")

    if args.extract:
        extract_videos(labeling, paths["aihub_root"], paths["raw_dir"] / "stage2_labeling")
        print(f"영상 → {paths['raw_dir'] / 'stage2_labeling'}")


if __name__ == "__main__":
    main()
