"""영상 → 세 Stage → 출력 JSON (영상 1개당 파일 1개). 처음부터 끝까지 도는 뼈대.

  # AI Hub mp4 폴더
  python scripts/run_inference.py --input data/raw/stage2_labeling --out outputs/baseline/aihub
  # comma2k19 test 분할 조각 (Stage 3 정밀 평가용)
  python scripts/run_inference.py --comma Chunk_1 --split test --out outputs/baseline/comma --limit 50

출력 형식은 docs/interfaces.md 2절. 쓰기 전에 형식 검사를 한다.
마지막에 영상당 평균 시간과 60분 안에 몇 개를 처리할 수 있는지 보여준다.
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cache import time_bins  # noqa: E402
from src.common.output import write_output  # noqa: E402
from src.common.paths import PROJECT_ROOT, load_paths  # noqa: E402
from src.common.video import comma_video_id, open_comma_segment, open_video  # noqa: E402
from src.pipeline import build_predictors, run_video  # noqa: E402


def aihub_jobs(input_dir: Path, limit: int | None):
    for p in sorted(input_dir.glob("*.mp4"))[:limit]:
        yield p.stem, (lambda p=p: open_video(p))


def comma_jobs(chunk: str, split: str, limit: int | None):
    paths = load_paths()
    labels = pd.read_csv(paths["comma_dir"] / "labels" / f"{chunk}.csv.gz", usecols=["segment", "split"])
    segs = labels.drop_duplicates("segment")
    if split != "all":
        segs = segs[segs["split"] == split]
    zpath = paths["comma_dir"] / "zips" / f"{chunk}.zip"
    work = paths["comma_dir"] / "work"
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    for seg in list(segs["segment"])[:limit]:
        full = f"{chunk}/{seg}"
        if f"{full}/video.hevc" not in names:
            continue
        yield comma_video_id(full), (lambda full=full: open_comma_segment(zpath, full, work))


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="mp4 폴더")
    src.add_argument("--comma", help="comma2k19 Chunk 이름 (예: Chunk_1)")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"], help="--comma 일 때")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    predictors = build_predictors(cfg)
    jobs = list(aihub_jobs(args.input, args.limit) if args.input else comma_jobs(args.comma, args.split, args.limit))
    if not jobs:
        sys.exit("처리할 영상 없음")

    t_start, totals, n_ok, total_video_sec = time.perf_counter(), {}, 0, 0.0
    for video_id, opener in tqdm(jobs, desc="추론"):
        t_open = time.perf_counter()
        with opener() as video:
            open_sec = time.perf_counter() - t_open
            doc = run_video(video, video_id, predictors)
            doc["runtime_sec"]["open"] = round(open_sec, 3)
            write_output(doc, args.out, n_bins=len(time_bins(video.duration_sec)))
            total_video_sec += video.duration_sec
        for k, v in doc["runtime_sec"].items():
            totals[k] = totals.get(k, 0.0) + v
        n_ok += 1

    wall = time.perf_counter() - t_start
    per_video = wall / n_ok
    print(f"\n예측기: {cfg['stage1']} / {cfg['stage2']} / {cfg['stage3']}  →  {args.out}")
    print(f"영상 {n_ok}개 ({total_video_sec / 60:.1f}분 분량), 전체 {wall:.1f}초, 영상당 {per_video:.2f}초")
    print("단계별 합계(초): " + ", ".join(f"{k} {v:.1f}" for k, v in totals.items()))
    print(f"이 속도면 {cfg['time_limit_min']}분 안에 약 {int(cfg['time_limit_min'] * 60 / per_video):,}개 처리 "
          "(이 PC 기준. 최종 확인은 L40S에서)")


if __name__ == "__main__":
    main()
