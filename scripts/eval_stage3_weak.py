"""Stage 3 예측을 AI Hub 약한 정답으로 평가한다.

실행:
  python scripts/eval_stage3_weak.py --pred-dir outputs/aihub     # 최종 출력 JSON 폴더 (<video_name>.json)
  python scripts/eval_stage3_weak.py --baseline straight          # 항상 '정속·직진' 베이스라인
  python scripts/eval_stage3_weak.py --baseline oracle            # 약한 정답을 그대로 낸 경우 (점검용, 100%가 나와야 함)

기본은 test 분할만 평가한다. 결과 표는 영상별 판정을 --save 경로에 CSV로 남긴다.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import PROJECT_ROOT, load_paths  # noqa: E402
from src.stage3_egomotion.weak_eval import WeakEvalConfig, evaluate  # noqa: E402

N_BINS = 100  # AI Hub 클립 10초 / 0.1초


def baseline_predictions(weak: pd.DataFrame, kind: str) -> dict[str, dict]:
    preds = {}
    for r in weak.itertuples(index=False):
        if kind == "straight":
            lat, lon = "직진", "정속"
        else:  # oracle
            lat = r.expected_lat
            lon = r.expected_long if isinstance(r.expected_long, str) else "정속"
        preds[r.video_name] = {"bin_sec": 0.1, "lat": [lat] * N_BINS, "long": [lon] * N_BINS}
    return preds


def load_predictions(pred_dir: Path) -> dict[str, dict]:
    preds = {}
    for p in pred_dir.glob("*.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        preds[doc.get("video_id", p.stem)] = doc["stage3"]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred-dir", type=Path)
    src.add_argument("--baseline", choices=["straight", "oracle"])
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--save", type=Path, help="영상별 판정 CSV 저장 경로")
    args = ap.parse_args()

    splits = load_paths()["splits_dir"]
    weak = pd.read_csv(splits / "stage3_weak.csv", encoding="utf-8-sig")
    if args.split != "all":
        weak = weak[weak["split_ours"] == args.split]
    check_path = splits / "stage3_weak_check.csv"
    check = pd.read_csv(check_path, encoding="utf-8-sig") if check_path.exists() else None

    y = yaml.safe_load((PROJECT_ROOT / "configs" / "stage3_weak.yaml").read_text(encoding="utf-8"))
    cfg = WeakEvalConfig(y["window_start_sec"], y["window_end_sec"], y["hit_ratio"])

    preds = baseline_predictions(weak, args.baseline) if args.baseline else load_predictions(args.pred_dir)
    report = evaluate(preds, weak, cfg, check)

    name = f"baseline={args.baseline}" if args.baseline else str(args.pred_dir)
    print(f"[{name}] split={args.split}  약한 정답 {report['n_weak']}건 중 예측 있음 {report['n_predicted']}건")
    print(f"판단 구간 {cfg.window_start_sec}~{cfg.window_end_sec}초, 기대 라벨 {cfg.hit_ratio:.0%} 이상이면 맞음")
    for axis, title in [("lat", "좌우"), ("long", "종방향(정지·출발 영상만)")]:
        if axis not in report:
            continue
        r = report[axis]
        print(f"\n{title}: 균형 정답률 {r['balanced_acc']:.1%}")
        for k, v in r["per_class"].items():
            print(f"  기대 {k:<3} {v['acc']:6.1%}  (n={v['n']})")
        print(f"  신뢰도별: {r['by_tier']}")
    if "weak_label_precision" in report:
        print("\n사람 점검으로 잰 약한 정답의 정확도 (점수 상한 참고):")
        for k, v in report["weak_label_precision"].items():
            print(f"  {k}: {v['precision']:.0%} (점검 {v['n_checked']}건)")
    else:
        print("\n사람 점검 결과 없음 → data/splits/stage3_weak_check.csv 의 check_result 를 채우면 정답 오류율이 함께 나온다")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        report["detail"].to_csv(args.save, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
