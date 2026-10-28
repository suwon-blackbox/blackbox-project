"""공통 코드가 docs/interfaces.md 규칙을 지키는지 확인한다.  실행: python -m pytest tests"""
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from src.common import cache
from src.common.paths import load_paths
from src.common.video import comma_video_id, open_comma_segment, open_video


@pytest.fixture
def synthetic_mp4(tmp_path):
    """15fps, 30프레임 영상. 프레임마다 밝기를 다르게 해 순서를 확인할 수 있게 한다."""
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, np.uint8))
    w.release()
    return path


def test_mp4_time_base(synthetic_mp4):
    v = open_video(synthetic_mp4)
    assert v.n_frames == 30
    assert v.fps == pytest.approx(15.0)
    assert v.duration_sec == pytest.approx(2.0)
    assert v.frame_at(0.0) == 0
    assert v.frame_at(1.0) == 15            # t=1.0초는 15번째 프레임
    assert v.frame_at(99.0) == 29           # 끝을 넘으면 마지막 프레임


def test_read_order_and_backward(synthetic_mp4):
    v = open_video(synthetic_mp4)
    later, earlier = v.read(20), v.read(3)  # 뒤로 가도 맞는 프레임
    assert later.mean() > earlier.mean()
    assert [i for i, _, _ in v.iter_frames(0.5, 1.0)] == list(range(8, 15))
    v.close()


def test_raw_stream_needs_times(tmp_path, synthetic_mp4):
    raw = tmp_path / "video.hevc"
    raw.write_bytes(synthetic_mp4.read_bytes())
    with pytest.raises(ValueError):
        open_video(raw)
    v = open_video(raw, fps=20.0)
    assert v.fps == pytest.approx(20.0)


def test_comma_video_id():
    assert comma_video_id("Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/10") == "b0c9d2329ad1606b_2018-07-27--06-03-57_10"


def test_comma_segment_is_20fps(tmp_path):
    zips = sorted((load_paths()["comma_dir"] / "zips").glob("Chunk_*.zip"))
    if not zips:
        pytest.skip("comma2k19 zip 없음")
    with zipfile.ZipFile(zips[0]) as z:
        seg = next(n for n in z.namelist() if n.endswith("/video.hevc")).rsplit("/", 1)[0]
    v = open_comma_segment(zips[0], seg, tmp_path)
    assert v.fps == pytest.approx(20.0, abs=0.05)   # OpenCV는 25로 잘못 읽는다
    assert v.n_frames == 1200


def test_cache_roundtrip(tmp_path, synthetic_mp4):
    v = open_video(synthetic_mp4)
    folder = cache.cache_path("aihub", "clip", root=tmp_path)
    meta = cache.write_meta(folder, v, "clip", "aihub", "train", "test")
    assert cache.read_meta(folder)["n_frames"] == meta["n_frames"] == 30

    tracks = pd.DataFrame({"frame": [0, 1], "t": [0.0, 1 / 15], "track_id": [1, 1], "cls": ["vehicle"] * 2,
                           "x": [1.0, 2.0], "y": [1.0, 1.0], "w": [10.0, 10.0], "h": [5.0, 5.0], "score": [0.9, 0.8]})
    cache.write_tracks(folder, tracks)
    assert len(cache.read_tracks(folder)) == 2

    t = cache.time_bins(v.duration_sec)
    assert len(t) == 20 and t[-1] == pytest.approx(1.9)
    k = len(t)
    cache.write_flow(folder, t=t, grid=np.zeros((k, 4, 4, 2)), divergence=np.zeros(k), curl=np.zeros(k),
                     mean_dx=np.zeros(k), valid=np.ones(k, bool))
    assert cache.read_flow(folder)["grid"].shape == (k, 4, 4, 2)


def test_cache_rejects_bad_format(tmp_path, synthetic_mp4):
    folder = cache.cache_path("aihub", "clip", root=tmp_path)
    cache.write_meta(folder, open_video(synthetic_mp4), "clip", "aihub", "train", "test")
    bad = pd.DataFrame({"frame": [0], "t": [0.0], "track_id": [1], "cls": ["car"],
                        "x": [0.0], "y": [0.0], "w": [1.0], "h": [1.0], "score": [1.0]})
    with pytest.raises(cache.CacheFormatError):
        cache.write_tracks(folder, bad)                       # 'car'는 허용 클래스 아님
    with pytest.raises(cache.CacheFormatError):
        cache.write_flow(folder, t=np.array([0.0, 0.2]), grid=np.zeros((2, 4, 4, 2)), divergence=np.zeros(2),
                         curl=np.zeros(2), mean_dx=np.zeros(2), valid=np.ones(2, bool))  # 간격 0.2초
    meta_path = folder / "meta.json"
    meta_path.write_text(meta_path.read_text(encoding="utf-8").replace('"cache_version": 0', '"cache_version": 99'),
                         encoding="utf-8")
    with pytest.raises(cache.CacheFormatError):
        cache.read_meta(folder)                               # 버전 다르면 거부
    with pytest.raises(ValueError):
        cache.cache_path("aihub", "route|seg", root=tmp_path)  # 폴더 이름에 | 금지
