"""AI Hub 약한 정답으로 Stage 3 예측을 영상 단위로 평가한다.

0.1초마다 맞았는지 보지 않는다. 충돌 직전 구간(기본 3.0~5.0초)에서 기대 라벨이
hit_ratio(기본 50%) 이상 나왔으면 그 영상은 맞음.

예측 형식은 docs/interfaces.md 의 최종 출력 JSON 중 stage3 부분:
  {"bin_sec": 0.1, "long": [...], "lat": [...]}
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WeakEvalConfig:
    window_start_sec: float = 3.0
    window_end_sec: float = 5.0
    hit_ratio: float = 0.5


def _window(stage3: dict, key: str, cfg: WeakEvalConfig) -> np.ndarray:
    labels = np.asarray(stage3[key])
    t = np.arange(len(labels)) * stage3.get("bin_sec", 0.1)
    return labels[(t >= cfg.window_start_sec) & (t < cfg.window_end_sec)]


def judge_video(stage3: dict, expected_lat, expected_long, cfg: WeakEvalConfig) -> dict:
    """영상 1개 판정. 기대값이 없는 축은 None."""
    out = {}
    for axis, expected in [("lat", expected_lat), ("long", expected_long)]:
        if expected is None or (isinstance(expected, float) and np.isnan(expected)):
            out[f"{axis}_hit"] = None
            out[f"{axis}_major"] = None
            continue
        w = _window(stage3, axis, cfg)
        if len(w) == 0:
            out[f"{axis}_hit"] = None
            out[f"{axis}_major"] = None
            continue
        values, counts = np.unique(w, return_counts=True)
        out[f"{axis}_major"] = values[counts.argmax()]
        out[f"{axis}_hit"] = bool((w == expected).mean() >= cfg.hit_ratio)
    return out


def evaluate(predictions: dict[str, dict], weak: pd.DataFrame, cfg: WeakEvalConfig | None = None,
             check: pd.DataFrame | None = None) -> dict:
    """predictions: video_name → stage3 dict.  weak: data/splits/stage3_weak.csv.

    check(사람 점검 결과)가 있으면 기대 라벨별 '약한 정답이 실제로 맞는 비율'을 함께 낸다.
    점수는 이 비율을 넘기 어렵다 (정답 자체가 그만큼 틀려 있으므로).
    """
    cfg = cfg or WeakEvalConfig()
    rows = []
    for r in weak.itertuples(index=False):
        stage3 = predictions.get(r.video_name)
        if stage3 is None:
            continue
        rows.append({"video_name": r.video_name, "tier": r.tier,
                     "expected_lat": r.expected_lat, "expected_long": r.expected_long,
                     **judge_video(stage3, r.expected_lat, r.expected_long, cfg)})
    res = pd.DataFrame(rows)
    report = {"n_weak": len(weak), "n_predicted": len(res)}
    if res.empty:
        return report

    for axis in ["lat", "long"]:
        sub = res[res[f"{axis}_hit"].notna()]
        if sub.empty:
            continue
        per_class = sub.groupby(f"expected_{axis}")[f"{axis}_hit"].agg(["mean", "size"])
        report[axis] = {
            # 클래스 수가 크게 달라서(직진이 대부분) 클래스별 정답률의 평균을 대표값으로 쓴다
            "balanced_acc": round(float(per_class["mean"].mean()), 3),
            "per_class": {k: {"acc": round(float(v["mean"]), 3), "n": int(v["size"])}
                          for k, v in per_class.iterrows()},
            "by_tier": {k: round(float(v), 3) for k, v in sub.groupby("tier")[f"{axis}_hit"].mean().items()},
            "confusion": pd.crosstab(sub[f"expected_{axis}"], sub[f"{axis}_major"]).to_dict(),
        }

    if check is not None:
        done = check[check["check_result"].isin(["yes", "no"])].copy()
        if len(done):
            done["expect"] = done["expected_lat"] + "·" + done["expected_long"].fillna("-")
            report["weak_label_precision"] = {
                k: {"precision": round(float((g["check_result"] == "yes").mean()), 3), "n_checked": len(g)}
                for k, g in done.groupby("expect")
            }
    report["detail"] = res
    return report
