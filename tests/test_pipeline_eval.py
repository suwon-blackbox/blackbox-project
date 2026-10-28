"""출력 형식 검사, 실행 뼈대, Stage 2·3 평가 계산 확인.  실행: python -m pytest tests"""
import copy

import cv2
import numpy as np
import pandas as pd
import pytest

from src.common.output import OutputFormatError, validate_output
from src.common.video import open_video
from src.pipeline import build_predictors, run_video
from src.stage2_timing import evaluate as s2
from src.stage3_egomotion import evaluate as s3


@pytest.fixture
def ten_sec_clip(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (32, 24))
    for _ in range(150):
        w.write(np.zeros((24, 32, 3), np.uint8))
    w.release()
    return path


@pytest.fixture
def baseline_doc(ten_sec_clip):
    preds = build_predictors({"stage1": "majority", "stage2": "middle", "stage3": "constant"})
    with open_video(ten_sec_clip) as v:
        return run_video(v, "clip", preds)


def test_baseline_output_is_valid(baseline_doc):
    validate_output(baseline_doc, n_bins=100)
    assert baseline_doc["stage2"]["collision"]["frame"] == 75      # 10초 클립 가운데
    assert len(baseline_doc["stage3"]["long"]) == 100


@pytest.mark.parametrize("breaker", [
    lambda d: d["stage3"]["lat"].pop(),                                   # 길이 불일치
    lambda d: d["stage3"].__setitem__("classes_lat", ["좌", "우", "직진"]),   # 클래스 순서
    lambda d: d["stage2"].__setitem__("entry_side", "LEFT"),
    lambda d: d["stage2"].__setitem__("entry", {"t": 9.0, "frame": 135, "confidence": 1}),  # 충돌보다 늦은 진입
    lambda d: d["stage1"].__setitem__("prob_rebroadcast", 1.5),
])
def test_validator_rejects(baseline_doc, breaker):
    doc = copy.deepcopy(baseline_doc)
    breaker(doc)
    with pytest.raises(OutputFormatError):
        validate_output(doc, n_bins=100)


def _gt(rows):
    cols = ["video_name", "collision_t", "entry_t", "entry_side", "avoidable_space", "quality"]
    return pd.DataFrame(rows, columns=cols)


def test_stage2_errors():
    gt = _gt([["a", 5.0, 3.0, "right", "no", "ok"],
              ["b", 4.0, None, "none", "yes", "ok"],
              ["c", 6.0, 5.0, "left", "unknown", "ok"]])
    ev = lambda t: {"t": t, "frame": 0, "confidence": 1}  # noqa: E731
    preds = {"a": {"collision": ev(5.1), "entry": ev(3.6), "entry_side": "right", "avoidable_space": "no"},
             "b": {"collision": ev(5.0), "entry": ev(2.0), "entry_side": "none", "avoidable_space": "no"},
             "c": {"collision": None, "entry": None, "entry_side": "none", "avoidable_space": "no"}}
    r = s2.evaluate(preds, gt)
    c = r["collision"]
    assert (c["n"], c["n_both"], c["missed"]) == (3, 2, 1)
    assert c["within_0.2s"] == pytest.approx(1 / 3, abs=1e-3)         # a만 맞음, 놓친 c도 분모에 포함
    assert c["mae_sec"] == pytest.approx((0.1 + 1.0) / 2, abs=1e-3)
    assert r["entry"]["false_alarm"] == 1                     # b는 진입이 없는데 예측함
    assert r["entry_side_acc"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["avoidable_n"] == 2 and r["avoidable_acc"] == pytest.approx(0.5)   # unknown 제외


def test_stage3_bins_align():
    labels = pd.DataFrame({"segment": ["r|x/1"] * 3, "bin": [1, 2, 3],   # bin 0은 센서가 없어 빠진 상황
                           "long_label": ["가속", "정속", "감속"], "lat_label": ["좌", "직진", "우"]})
    pred = {"long": ["감속", "가속", "정속", "감속"], "lat": ["우", "좌", "직진", "우"]}
    r = s3.evaluate({"r_x_1": pred}, labels)
    assert r["n_bins"] == 3 and r["long"]["acc"] == 1.0 and r["lat"]["acc"] == 1.0   # 한 칸 밀리면 0이 된다
    assert r["combo9"]["balanced_acc"] == 1.0
