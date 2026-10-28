"""최종 출력 JSON (docs/interfaces.md 2절) 만들기와 검사."""
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 0
CLASSES_LONG = ["가속", "감속", "정속"]
CLASSES_LAT = ["좌", "직진", "우"]
ENTRY_SIDES = {"left", "right", "front", "rear", "none"}
AVOIDABLE = {"yes", "no", "unknown"}
STAGE1_LABELS = {"original", "rebroadcast"}


class OutputFormatError(ValueError):
    pass


def validate_output(doc: dict, n_bins: int | None = None) -> None:
    """형식이 틀리면 OutputFormatError. n_bins를 주면 Stage 3 길이도 확인."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise OutputFormatError(f"{doc.get('video_id', '?')}: {msg}")

    need(doc.get("schema_version") == SCHEMA_VERSION, f"schema_version ≠ {SCHEMA_VERSION}")
    for key in ["video_id", "fps", "duration_sec", "stage1", "stage2", "stage3", "runtime_sec"]:
        need(key in doc, f"'{key}' 없음")

    s1 = doc["stage1"]
    need(s1.get("label") in STAGE1_LABELS, f"stage1.label은 {STAGE1_LABELS} 중 하나")
    need(0.0 <= float(s1.get("prob_rebroadcast", -1)) <= 1.0, "stage1.prob_rebroadcast는 0~1")

    s2 = doc["stage2"]
    for ev in ["collision", "entry"]:
        e = s2.get(ev, "missing")
        need(e != "missing", f"stage2.{ev} 없음 (없으면 null)")
        if e is not None:
            need({"t", "frame", "confidence"} <= set(e), f"stage2.{ev}에는 t, frame, confidence")
            need(0 <= e["t"] <= doc["duration_sec"], f"stage2.{ev}.t가 영상 길이 밖")
    need(s2.get("entry_side") in ENTRY_SIDES, f"stage2.entry_side는 {ENTRY_SIDES} 중 하나")
    need(s2.get("avoidable_space") in AVOIDABLE, f"stage2.avoidable_space는 {AVOIDABLE} 중 하나")
    if s2.get("collision") and s2.get("entry"):
        need(s2["entry"]["t"] <= s2["collision"]["t"], "진입이 충돌보다 늦음")

    s3 = doc["stage3"]
    need(s3.get("bin_sec") == 0.1, "stage3.bin_sec는 0.1")
    need(s3.get("classes_long") == CLASSES_LONG and s3.get("classes_lat") == CLASSES_LAT,
         f"stage3 클래스 순서는 long {CLASSES_LONG}, lat {CLASSES_LAT}")
    k = len(s3["long"])
    need(len(s3["lat"]) == k, "stage3.long과 lat 길이가 다름")
    if n_bins is not None:
        need(k == n_bins, f"stage3 길이 {k} ≠ {n_bins} (영상 길이 / 0.1초)")
    need(set(s3["long"]) <= set(CLASSES_LONG) and set(s3["lat"]) <= set(CLASSES_LAT), "stage3에 알 수 없는 클래스")
    for key in ["p_long", "p_lat"]:
        if key in s3:
            p = np.asarray(s3[key], dtype=float)
            need(p.shape == (k, 3), f"stage3.{key} shape {p.shape} ≠ ({k}, 3)")
            need(np.allclose(p.sum(axis=1), 1.0, atol=1e-3), f"stage3.{key} 각 행의 합은 1")


def write_output(doc: dict, out_dir: Path, n_bins: int | None = None) -> Path:
    validate_output(doc, n_bins)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc['video_id']}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def read_outputs(out_dir: Path) -> dict[str, dict]:
    docs = {}
    for p in sorted(Path(out_dir).glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        validate_output(doc)
        docs[doc["video_id"]] = doc
    return docs
