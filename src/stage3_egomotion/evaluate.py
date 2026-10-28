"""Stage 3 정밀 평가: comma2k19 센서 정답(0.1초 구간)과 예측을 구간별로 비교한다.

예측 배열의 k번째 = 구간 k = [0.1k, 0.1k+0.1)초 = 정답 CSV의 bin == k.
센서가 없어 정답이 빠진 구간은 계산에서 뺀다.
"""
import numpy as np
import pandas as pd

from src.common.output import CLASSES_LAT, CLASSES_LONG
from src.common.video import comma_video_id


def per_class(gt: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    out = {}
    for c in classes:
        m = gt == c
        out[c] = {"recall": round(float((pred[m] == c).mean()), 3) if m.any() else None, "n": int(m.sum())}
    return out


def evaluate(predictions: dict[str, dict], labels: pd.DataFrame) -> dict:
    """predictions: video_id(comma_video_id) → stage3 dict.  labels: Chunk_N.csv.gz 를 합친 것."""
    labels = labels.copy()
    labels["video_id"] = labels["segment"].map(comma_video_id)
    gt_long, gt_lat, pr_long, pr_lat = [], [], [], []
    n_videos = 0
    for vid, g in labels.groupby("video_id"):
        p = predictions.get(vid)
        if p is None:
            continue
        n_videos += 1
        k = g["bin"].to_numpy()
        ok = k < len(p["long"])
        gt_long.append(g["long_label"].to_numpy()[ok]); gt_lat.append(g["lat_label"].to_numpy()[ok])  # noqa: E702
        pr_long.append(np.asarray(p["long"])[k[ok]]); pr_lat.append(np.asarray(p["lat"])[k[ok]])  # noqa: E702
    if not n_videos:
        return {"n_videos": 0}
    gl, ga, pl, pa = map(np.concatenate, (gt_long, gt_lat, pr_long, pr_lat))
    long_pc, lat_pc = per_class(gl, pl, CLASSES_LONG), per_class(ga, pa, CLASSES_LAT)
    combo_gt, combo_pr = np.char.add(np.char.add(gl.astype(str), "·"), ga.astype(str)), \
        np.char.add(np.char.add(pl.astype(str), "·"), pa.astype(str))
    combos = sorted(set(combo_gt))
    combo_recall = [float((combo_pr[combo_gt == c] == c).mean()) for c in combos]

    def balanced(pc: dict) -> float:
        vals = [v["recall"] for v in pc.values() if v["recall"] is not None]
        return round(float(np.mean(vals)), 3)

    return {
        "n_videos": n_videos, "n_bins": int(len(gl)), "hours": round(len(gl) * 0.1 / 3600, 2),
        "long": {"acc": round(float((gl == pl).mean()), 3), "balanced_acc": balanced(long_pc), "per_class": long_pc,
                 "confusion": pd.crosstab(pd.Series(gl, name="gt"), pd.Series(pl, name="pred"))},
        "lat": {"acc": round(float((ga == pa).mean()), 3), "balanced_acc": balanced(lat_pc), "per_class": lat_pc,
                "confusion": pd.crosstab(pd.Series(ga, name="gt"), pd.Series(pa, name="pred"))},
        "combo9": {"acc": round(float((combo_gt == combo_pr).mean()), 3),
                   "balanced_acc": round(float(np.mean(combo_recall)), 3)},
    }
