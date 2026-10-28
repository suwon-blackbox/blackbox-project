"""Stage 2 평가.

  python scripts/eval_stage2.py --pred-dir outputs/baseline/aihub            # 예측 vs 사람 라벨
  python scripts/eval_stage2.py --agreement                                  # 두 라벨러 일치도 (30건)

정답: data/labels/stage2/*.csv (scripts/label_stage2.py 결과). 기본은 test 분할만.
라벨이 아직 적을 때는 --split all 로 전체를 본다 (그때 숫자는 참고용).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.output import read_outputs  # noqa: E402
from src.common.paths import load_paths  # noqa: E402
from src.stage2_timing.evaluate import agreement, evaluate, load_ground_truth  # noqa: E402


def show_event(name: str, r: dict) -> None:
    if not r.get("n_both"):
        print(f"  {name}: 정답 {r['n']}건, 비교 가능 0건")
        return
    print(f"  {name}: 정답 {r['n']}건 | 0.2초 이내 {r['within_0.2s']:.1%} | 0.5초 이내 {r['within_0.5s']:.1%} | "
          f"평균 오차 {r['mae_sec']:.2f}초 | 놓침 {r['missed']} | 헛예측 {r['false_alarm']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred-dir", type=Path)
    mode.add_argument("--agreement", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = ap.parse_args()

    paths = load_paths()
    files = sorted((paths["splits_dir"].parent / "labels" / "stage2").glob("*.csv"))
    if not files:
        sys.exit("라벨 파일 없음 → python scripts/label_stage2.py --annotator <이름>")
    lab_list = pd.read_csv(paths["splits_dir"] / "stage2_labeling.csv", encoding="utf-8-sig")

    if args.agreement:
        r = agreement(files, lab_list)
        print(f"공동 라벨링 {r['n_shared_done']}/{r['n_shared_total']}건 완료")
        if r["n_shared_done"]:
            print(f"  충돌 시점 0.2초 이내 일치 {r['collision_within_0.2s']:.0%} (목표 90% 이상), "
                  f"차이 중앙값 {r['collision_median_diff_sec']}초")
            print(f"  진입 시점 차이 중앙값 {r['entry_median_diff_sec']}초")
            print(f"  진입 방향 일치 {r['side_agree']:.0%}, 피할 공간 일치 {r['avoid_agree']:.0%}, "
                  f"품질 일치 {r['quality_agree']:.0%}")
        return

    gt = load_ground_truth(files, lab_list)
    if args.split != "all":
        gt = gt[gt["split_ours"] == args.split]
    preds = {vid: doc["stage2"] for vid, doc in read_outputs(args.pred_dir).items()}
    r = evaluate(preds, gt)
    print(f"[{args.pred_dir}] split={args.split}  정답 {r['n_gt']}건 중 예측 있음 {r['n_predicted']}건")
    if r["n_predicted"] == 0:
        return
    show_event("충돌", r["collision"])
    show_event("진입", r["entry"])
    print(f"  진입 방향 정확도 {r['entry_side_acc']:.1%}")
    if "avoidable_acc" in r:
        print(f"  피할 공간 정확도 {r['avoidable_acc']:.1%} (yes/no {r['avoidable_n']}건)")
    if r["n_gt"] < 30:
        print("  ※ 정답이 30건 미만이라 숫자가 크게 흔들린다 (참고용)")


if __name__ == "__main__":
    main()
