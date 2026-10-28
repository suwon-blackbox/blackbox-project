"""Stage 3 정밀 평가 (comma2k19 센서 정답).

  python scripts/run_inference.py --comma Chunk_1 --split test --out outputs/baseline/comma
  python scripts/eval_stage3.py --pred-dir outputs/baseline/comma

정답은 data/comma2k19/labels/Chunk_*.csv.gz. 클래스 불균형이 커서(정속·직진이 대부분)
균형 정답률(클래스별 재현율의 평균)을 대표값으로 본다.
한국 영상에서의 확인은 scripts/eval_stage3_weak.py (AI Hub 약한 정답).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.output import read_outputs  # noqa: E402
from src.common.paths import load_paths  # noqa: E402
from src.stage3_egomotion.evaluate import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = ap.parse_args()

    files = sorted((load_paths()["comma_dir"] / "labels").glob("Chunk_*.csv.gz"))
    labels = pd.concat([pd.read_csv(f, usecols=["segment", "split", "bin", "long_label", "lat_label"]) for f in files])
    if args.split != "all":
        labels = labels[labels["split"] == args.split]
    preds = {vid: doc["stage3"] for vid, doc in read_outputs(args.pred_dir).items()}
    r = evaluate(preds, labels)
    if not r["n_videos"]:
        sys.exit("예측과 맞는 조각이 없음 (video_id 확인)")

    print(f"[{args.pred_dir}] split={args.split}  조각 {r['n_videos']}개, {r['hours']}시간, 구간 {r['n_bins']:,}개")
    for axis, title in [("long", "종방향"), ("lat", "좌우")]:
        a = r[axis]
        recalls = ", ".join(f"{c} {v['recall']:.0%}(n={v['n']:,})" for c, v in a["per_class"].items() if v["recall"] is not None)
        print(f"  {title}: 균형 정답률 {a['balanced_acc']:.1%} (단순 정답률 {a['acc']:.1%}) | {recalls}")
    print(f"  9개 조합: 균형 정답률 {r['combo9']['balanced_acc']:.1%} (단순 정답률 {r['combo9']['acc']:.1%})")


if __name__ == "__main__":
    main()
