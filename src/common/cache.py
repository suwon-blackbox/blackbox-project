"""공통 전처리 캐시 읽기·쓰기 (docs/interfaces.md 1절).

폴더: <cache_dir>/<source>/<video_id>/{meta.json, tracks.csv, flow.npz}
형식을 바꾸면 CACHE_VERSION 을 올리고 docs/interfaces.md 를 같이 고친다.
읽을 때 버전이나 형식이 다르면 바로 에러를 내서, 서로 다른 형식이 조용히 섞이지 않게 한다.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, load_paths
from src.common.video import Video

CACHE_VERSION = 0
SOURCES = ("aihub", "comma2k19", "recapture")
TRACK_COLUMNS = ["frame", "t", "track_id", "cls", "x", "y", "w", "h", "score"]
TRACK_CLASSES = {"vehicle", "two_wheeler", "bicycle", "pedestrian"}
FLOW_KEYS = {"t", "grid", "divergence", "curl", "mean_dx", "valid"}
FLOW_BIN_SEC = 0.1


class CacheFormatError(ValueError):
    pass


def cache_path(source: str, video_id: str, root: Path | None = None) -> Path:
    if source not in SOURCES:
        raise ValueError(f"source는 {SOURCES} 중 하나: {source}")
    if "|" in video_id or "/" in video_id:
        raise ValueError(f"video_id에 | 나 / 를 쓸 수 없음: {video_id}")
    return (root or load_paths()["cache_dir"]) / source / video_id


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------- meta.json ----------

def write_meta(folder: Path, video: Video, video_id: str, source: str, split_ours: str, created_by: str) -> dict:
    meta = {
        "cache_version": CACHE_VERSION,
        "video_id": video_id,
        "source": source,
        "split_ours": split_ours,
        "fps": round(video.fps, 4),
        "n_frames": video.n_frames,
        "duration_sec": round(video.duration_sec, 4),
        "width": video.width,
        "height": video.height,
        "created_by": f"{created_by}@{_git_commit()}",
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def read_meta(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta.get("cache_version") != CACHE_VERSION:
        raise CacheFormatError(f"{folder}: 캐시 버전 {meta.get('cache_version')} ≠ 코드 버전 {CACHE_VERSION} → 다시 생성")
    return meta


# ---------- tracks.csv ----------

def write_tracks(folder: Path, tracks: pd.DataFrame) -> None:
    validate_tracks(tracks)
    tracks[TRACK_COLUMNS].to_csv(folder / "tracks.csv", index=False)


def read_tracks(folder: Path) -> pd.DataFrame:
    read_meta(folder)
    tracks = pd.read_csv(folder / "tracks.csv")
    validate_tracks(tracks)
    return tracks


def validate_tracks(tracks: pd.DataFrame) -> None:
    missing = [c for c in TRACK_COLUMNS if c not in tracks.columns]
    if missing:
        raise CacheFormatError(f"tracks에 없는 열: {missing}")
    bad = set(tracks["cls"].unique()) - TRACK_CLASSES
    if bad:
        raise CacheFormatError(f"알 수 없는 cls: {bad} (허용: {sorted(TRACK_CLASSES)})")
    if (tracks[["w", "h"]] <= 0).any().any():
        raise CacheFormatError("bbox 너비·높이는 양수여야 함 ([x, y, w, h] 좌상단 기준)")


# ---------- flow.npz ----------

def write_flow(folder: Path, **arrays: np.ndarray) -> None:
    validate_flow(arrays)
    np.savez_compressed(folder / "flow.npz", **arrays)


def read_flow(folder: Path) -> dict[str, np.ndarray]:
    read_meta(folder)
    with np.load(folder / "flow.npz") as f:
        arrays = {k: f[k] for k in f.files}
    validate_flow(arrays)
    return arrays


def validate_flow(arrays: dict[str, np.ndarray]) -> None:
    missing = FLOW_KEYS - set(arrays)
    if missing:
        raise CacheFormatError(f"flow에 없는 키: {sorted(missing)}")
    k = len(arrays["t"])
    expected = {"grid": (k, 4, 4, 2), "divergence": (k,), "curl": (k,), "mean_dx": (k,), "valid": (k,)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise CacheFormatError(f"flow['{key}'] shape {arrays[key].shape} ≠ {shape}")
    if k > 1 and not np.allclose(np.diff(arrays["t"]), FLOW_BIN_SEC, atol=1e-6):
        raise CacheFormatError(f"flow['t'] 간격은 {FLOW_BIN_SEC}초여야 함")


def time_bins(duration_sec: float, bin_sec: float = FLOW_BIN_SEC) -> np.ndarray:
    """0.1초 구간 시작 시각 0.0, 0.1, … (마지막 구간이 영상 끝을 넘지 않게)."""
    k = int(np.floor(duration_sec / bin_sec + 1e-9))
    return np.round(np.arange(k) * bin_sec, 6)
