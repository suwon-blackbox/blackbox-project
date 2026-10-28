"""영상 읽기. 모든 Stage가 이 모듈로 영상을 연다.

핵심 규칙 (docs/interfaces.md 0절): 시점은 초 단위 t. 프레임 번호는 원본 기준 0부터.

주의: comma2k19의 video.hevc는 컨테이너가 없는 날 스트림이라 OpenCV가 fps를 25로 잘못 읽는다
(실제 20fps). 그래서 comma 영상은 반드시 global_pose/frame_times 로 시각을 준다.
"""
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Video:
    path: Path
    times: np.ndarray            # 프레임별 시각(초), times[0] == 0.0
    width: int
    height: int
    _cap: cv2.VideoCapture | None = field(default=None, repr=False)
    _next_idx: int = field(default=0, repr=False)

    @property
    def n_frames(self) -> int:
        return len(self.times)

    @property
    def fps(self) -> float:
        return float(1.0 / np.median(np.diff(self.times))) if self.n_frames > 1 else 0.0

    @property
    def duration_sec(self) -> float:
        return float(self.times[-1] + 1.0 / self.fps) if self.n_frames > 1 else 0.0

    def frame_at(self, t: float) -> int:
        """시각 t를 포함하는 프레임 번호 (t 이전의 마지막 프레임)."""
        return int(np.clip(np.searchsorted(self.times, t + 1e-9, side="right") - 1, 0, self.n_frames - 1))

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.path))
            self._next_idx = 0
        return self._cap

    def read(self, idx: int) -> np.ndarray:
        """프레임 하나(BGR). 순서대로 읽으면 빠르고, 뒤로 가면 처음부터 다시 연다.

        날 HEVC 스트림은 seek가 부정확해서 위치 이동(CAP_PROP_POS_FRAMES)을 쓰지 않는다.
        """
        if not 0 <= idx < self.n_frames:
            raise IndexError(f"frame {idx} 범위 밖 (0~{self.n_frames - 1})")
        cap = self._ensure_open()
        if idx < self._next_idx:
            self.close()
            cap = self._ensure_open()
        frame = None
        while self._next_idx <= idx:
            ok, frame = cap.read()
            if not ok:
                raise IOError(f"{self.path}: frame {self._next_idx} 디코딩 실패")
            self._next_idx += 1
        return frame

    def iter_frames(self, start_t: float = 0.0, end_t: float | None = None,
                    step: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
        """[start_t, end_t) 안의 (프레임 번호, 시각, 프레임)을 순서대로. step=2면 한 장 건너 하나."""
        end_t = self.duration_sec if end_t is None else end_t
        first = int(np.searchsorted(self.times, start_t - 1e-9, side="left"))
        for idx in range(first, self.n_frames, step):
            t = float(self.times[idx])
            if t >= end_t:
                break
            yield idx, t, self.read(idx)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._next_idx = 0

    def __enter__(self) -> "Video":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_video(path: str | Path, frame_times: np.ndarray | None = None, fps: float | None = None) -> Video:
    """영상 열기.

    - frame_times: 프레임별 시각(초). comma2k19는 필수 (global_pose/frame_times)
    - fps: 시각 정보가 없을 때 직접 지정
    - 둘 다 없으면 컨테이너의 fps를 쓴다 (mp4 등). 날 .hevc/.h264 는 에러
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없음: {path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    container_fps = cap.get(cv2.CAP_PROP_FPS)
    container_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_times is not None:
        times = np.asarray(frame_times, dtype=np.float64)
        times = times - times[0]
    else:
        if fps is None:
            if path.suffix.lower() in {".hevc", ".h265", ".h264", ".264"}:
                raise ValueError(f"{path.name}: 날 스트림은 fps를 믿을 수 없음 → frame_times 또는 fps를 지정")
            fps = container_fps
        if not fps or fps <= 0:
            raise ValueError(f"{path.name}: fps를 알 수 없음")
        n = container_n if container_n > 0 else _count_frames(path)
        times = np.arange(n) / fps
    return Video(path=path, times=times, width=width, height=height)


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


# ---------- comma2k19 ----------

def comma_video_id(segment: str) -> str:
    """'Chunk_1/b0c9...|2018-07-27--06-03-57/10' → 'b0c9..._2018-07-27--06-03-57_10' (폴더 이름에 | 불가)."""
    route, seg = segment.rstrip("/").split("/")[-2:]
    return f"{route.replace('|', '_')}_{seg}"


def extract_comma_segment(zip_path: str | Path, segment: str, out_dir: str | Path) -> tuple[Path, np.ndarray]:
    """zip 안의 1분 조각에서 video.hevc 와 frame_times 를 꺼낸다. 이미 있으면 다시 꺼내지 않는다.

    segment: 'Chunk_1/<route>/<seg>' (zip 안 경로, 끝의 / 없이)
    반환: (video.hevc 경로, frame_times)
    """
    out = Path(out_dir) / comma_video_id(segment)
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / "video.hevc"
    times_path = out / "frame_times.npy"
    if not (video_path.exists() and times_path.exists()):
        with zipfile.ZipFile(zip_path) as z:
            video_path.write_bytes(z.read(f"{segment}/video.hevc"))
            times = np.load(io.BytesIO(z.read(f"{segment}/global_pose/frame_times")))
            np.save(times_path, times)
    return video_path, np.load(times_path)


def open_comma_segment(zip_path: str | Path, segment: str, work_dir: str | Path) -> Video:
    video_path, times = extract_comma_segment(zip_path, segment, work_dir)
    return open_video(video_path, frame_times=times)
