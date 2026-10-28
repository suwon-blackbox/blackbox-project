"""Stage 2 평가: 사람이 단 시점 라벨(data/labels/stage2/*.csv)과 예측 비교.

시점은 초 단위로 비교한다 (fps가 달라도 같은 기준). 기본 허용 오차 0.2초 = AI Hub 3프레임.
"""
import numpy as np
import pandas as pd

TOLERANCES = (0.2, 0.5)            # 초. 0.2초 = 15fps에서 3프레임
EXCLUDE_QUALITY = {"bad_video"}


def load_ground_truth(label_files: list, labeling_list: pd.DataFrame) -> pd.DataFrame:
    """라벨 파일들을 합쳐 영상당 1행. 두 사람이 다 단 영상은 annotator_1 것을 정답으로 쓴다."""
    labels = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in label_files], ignore_index=True)
    first = labeling_list.set_index("video_name")["annotator_1"]
    labels["is_primary"] = labels["annotator"] == labels["video_name"].map(first)
    gt = labels.sort_values("is_primary", ascending=False).drop_duplicates("video_name")
    gt = gt[~gt["quality"].isin(EXCLUDE_QUALITY)]
    split = labeling_list.set_index("video_name")["split_ours"]
    gt["split_ours"] = gt["video_name"].map(split)
    return gt.reset_index(drop=True)


def _event_errors(gt_t: pd.Series, pred_t: pd.Series) -> dict:
    both = gt_t.notna() & pred_t.notna()
    err = (pred_t[both] - gt_t[both]).abs()
    out = {"n": int(gt_t.notna().sum()), "n_both": int(both.sum()),
           # 정답은 있는데 예측이 없음 / 정답은 없는데 예측함
           "missed": int((gt_t.notna() & pred_t.isna()).sum()),
           "false_alarm": int((gt_t.isna() & pred_t.notna()).sum())}
    if len(err):
        out["mae_sec"] = round(float(err.mean()), 3)
        out["median_sec"] = round(float(err.median()), 3)
        for tol in TOLERANCES:
            # 정답이 있는 영상 전체 대비 비율 (예측을 안 한 것도 틀린 것으로 센다)
            out[f"within_{tol}s"] = round(float((err <= tol + 1e-9).sum() / max(1, out["n"])), 3)
    return out


def evaluate(predictions: dict[str, dict], gt: pd.DataFrame) -> dict:
    """predictions: video_name → 출력 JSON의 stage2 dict."""
    rows = []
    for r in gt.itertuples(index=False):
        p = predictions.get(r.video_name)
        if p is None:
            continue
        rows.append({
            "video_name": r.video_name,
            "gt_collision_t": r.collision_t, "pred_collision_t": (p["collision"] or {}).get("t"),
            "gt_entry_t": r.entry_t, "pred_entry_t": (p["entry"] or {}).get("t"),
            "gt_side": r.entry_side, "pred_side": p["entry_side"],
            "gt_avoid": r.avoidable_space, "pred_avoid": p["avoidable_space"],
        })
    res = pd.DataFrame(rows)
    report = {"n_gt": len(gt), "n_predicted": len(res)}
    if res.empty:
        return report
    for col in ["gt_collision_t", "pred_collision_t", "gt_entry_t", "pred_entry_t"]:
        res[col] = pd.to_numeric(res[col], errors="coerce")
    report["collision"] = _event_errors(res["gt_collision_t"], res["pred_collision_t"])
    report["entry"] = _event_errors(res["gt_entry_t"], res["pred_entry_t"])
    report["entry_side_acc"] = round(float((res["gt_side"] == res["pred_side"]).mean()), 3)
    known = res[res["gt_avoid"].isin(["yes", "no"])]
    if len(known):
        report["avoidable_acc"] = round(float((known["gt_avoid"] == known["pred_avoid"]).mean()), 3)
        report["avoidable_n"] = len(known)
    report["detail"] = res
    return report


def agreement(label_files: list, labeling_list: pd.DataFrame) -> dict:
    """두 사람이 같이 단 영상에서 서로 얼마나 일치하는지 (기준표가 명확한지 확인용)."""
    labels = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in label_files], ignore_index=True)
    shared = labeling_list[labeling_list["annotator_2"].notna()]
    rows = []
    for r in shared.itertuples(index=False):
        a = labels[(labels["video_name"] == r.video_name) & (labels["annotator"] == r.annotator_1)]
        b = labels[(labels["video_name"] == r.video_name) & (labels["annotator"] == r.annotator_2)]
        if len(a) and len(b):
            a, b = a.iloc[0], b.iloc[0]
            rows.append({"video_name": r.video_name,
                         "collision_diff_sec": abs(a["collision_t"] - b["collision_t"])
                         if pd.notna(a["collision_t"]) and pd.notna(b["collision_t"]) else np.nan,
                         "entry_diff_sec": abs(a["entry_t"] - b["entry_t"])
                         if pd.notna(a["entry_t"]) and pd.notna(b["entry_t"]) else np.nan,
                         "side_same": a["entry_side"] == b["entry_side"],
                         "avoid_same": a["avoidable_space"] == b["avoidable_space"],
                         "quality_same": a["quality"] == b["quality"]})
    res = pd.DataFrame(rows)
    if res.empty:
        return {"n_shared_done": 0, "n_shared_total": len(shared)}
    c = res["collision_diff_sec"].dropna()
    return {
        "n_shared_done": len(res), "n_shared_total": len(shared),
        "collision_within_0.2s": round(float((c <= 0.2 + 1e-9).mean()), 3) if len(c) else None,
        "collision_median_diff_sec": round(float(c.median()), 3) if len(c) else None,
        "entry_median_diff_sec": round(float(res["entry_diff_sec"].dropna().median()), 3)
        if res["entry_diff_sec"].notna().any() else None,
        "side_agree": round(float(res["side_same"].mean()), 3),
        "avoid_agree": round(float(res["avoid_same"].mean()), 3),
        "quality_agree": round(float(res["quality_same"].mean()), 3),
        "detail": res,
    }
