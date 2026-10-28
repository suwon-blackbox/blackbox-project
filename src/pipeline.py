"""영상 1개 → 세 Stage → 최종 출력 JSON.

각 Stage는 predict()만 맞추면 갈아 끼울 수 있다. 지금은 모두 베이스라인이고,
모델이 준비되면 PREDICTORS에 등록하고 configs/pipeline.yaml 에서 이름만 바꾼다.

    Stage1Predictor.predict(video)            -> {"label", "prob_rebroadcast"}
    Stage2Predictor.predict(video, cache_dir) -> {"collision", "entry", "entry_side", "avoidable_space", "opponent"}
    Stage3Predictor.predict(video, cache_dir) -> {"long": [...], "lat": [...], "p_long": [...], "p_lat": [...]}
"""
import time
from pathlib import Path

from src.common.cache import time_bins
from src.common.output import CLASSES_LAT, CLASSES_LONG, SCHEMA_VERSION
from src.common.video import Video


class Stage1Majority:
    """베이스라인: 항상 원본 (재촬영 데이터가 생기면 다수 클래스로 바꾼다)."""
    def predict(self, video: Video) -> dict:
        return {"label": "original", "prob_rebroadcast": 0.0}


class Stage2Middle:
    """베이스라인: 충돌 = 영상 가운데. AI Hub 10초 클립이면 5.0초(75프레임). 진입 없음."""
    def predict(self, video: Video, cache_dir: Path | None) -> dict:
        t = video.duration_sec / 2
        frame = video.frame_at(t)
        return {"collision": {"t": round(float(video.times[frame]), 4), "frame": frame, "confidence": 0.0},
                "entry": None, "entry_side": "none", "avoidable_space": "unknown", "opponent": None}


class Stage3Constant:
    """베이스라인: 항상 정속·직진."""
    def predict(self, video: Video, cache_dir: Path | None) -> dict:
        k = len(time_bins(video.duration_sec))
        one_hot = lambda classes, c: [1.0 if x == c else 0.0 for x in classes]  # noqa: E731
        return {"long": ["정속"] * k, "lat": ["직진"] * k,
                "p_long": [one_hot(CLASSES_LONG, "정속")] * k, "p_lat": [one_hot(CLASSES_LAT, "직진")] * k}


PREDICTORS = {
    "stage1": {"majority": Stage1Majority},
    "stage2": {"middle": Stage2Middle},
    "stage3": {"constant": Stage3Constant},
}


def preprocess(video: Video, cache_dir: Path | None) -> None:
    """공통 전처리 자리 (B 담당). 검출·추적·광류 캐시를 만들거나, 이미 있으면 그대로 둔다."""
    return None


def build_predictors(cfg: dict) -> dict:
    return {stage: PREDICTORS[stage][cfg[stage]]() for stage in ("stage1", "stage2", "stage3")}


def run_video(video: Video, video_id: str, predictors: dict, cache_dir: Path | None = None) -> dict:
    runtime = {}

    def timed(name, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        runtime[name] = round(time.perf_counter() - t0, 3)
        return out

    timed("preprocess", preprocess, video, cache_dir)
    s1 = timed("stage1", predictors["stage1"].predict, video)
    s2 = timed("stage2", predictors["stage2"].predict, video, cache_dir)
    s3 = timed("stage3", predictors["stage3"].predict, video, cache_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "fps": round(video.fps, 4),
        "duration_sec": round(video.duration_sec, 4),
        "stage1": s1,
        "stage2": s2,
        "stage3": {"bin_sec": 0.1, "classes_long": CLASSES_LONG, "classes_lat": CLASSES_LAT, **s3},
        "runtime_sec": runtime,
    }
