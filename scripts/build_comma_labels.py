"""comma2k19 센서 기록으로 Stage 3 정답(0.1초마다 가속/감속/정속 × 좌/우/직진)을 만든다.

zip을 풀지 않고 각 1분 조각(segment)의 numpy 배열만 읽는다. 영상(video.hevc)은 건드리지 않는다.

확인한 데이터 사실 (Chunk_1 기준)
  - 영상 20fps, 조각당 1200프레임(60초). global_pose/frame_times 가 프레임별 시각
  - CAN speed 약 83Hz (m/s), steering_angle 약 83Hz (deg, +가 좌회전)
  - IMU gyro 20Hz, 축 [전방, 우측, 아래] → z(+)가 우회전. 조향각과 상관 -0.7 ~ -0.99로 부호 확인

출력: comma_dir/labels/<Chunk_N>.csv.gz  — 0.1초 구간 1개당 1행
  segment, route, split, bin(=구간 번호, t=0.1·bin), t(구간 시작 초), frame_start, speed, accel, yaw_rate_dps, steering_deg, long_label, lat_label

실행:
  python scripts/build_comma_labels.py                    # zips/ 안의 모든 Chunk
  python scripts/build_comma_labels.py --chunks Chunk_1
"""
import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cache import time_bins  # noqa: E402
from src.common.paths import PROJECT_ROOT, load_paths  # noqa: E402

LONG_NAMES = np.array(["감속", "정속", "가속"])
LAT_NAMES = np.array(["좌", "직진", "우"])


def route_split(route: str, test_percent: int) -> str:
    h = int(hashlib.md5(route.encode("utf-8")).hexdigest(), 16)
    return "test" if h % 100 < test_percent else "train"


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")


def segment_labels(z: zipfile.ZipFile, seg: str, cfg: dict) -> pd.DataFrame:
    def load(rel: str) -> np.ndarray:
        return np.load(io.BytesIO(z.read(f"{seg}/{rel}")))

    frame_t = load("global_pose/frame_times")
    sp_t, sp = load("processed_log/CAN/speed/t"), load("processed_log/CAN/speed/value").ravel()
    st_t, st = load("processed_log/CAN/steering_angle/t"), load("processed_log/CAN/steering_angle/value").ravel()
    g_t, gyro = load("processed_log/IMU/gyro/t"), load("processed_log/IMU/gyro/value")

    # 구간은 시간으로 정한다 (docs/interfaces.md): 구간 k = [0.1k, 0.1k+0.1)초, 값은 구간 중앙에서 잰다.
    # 센서가 없는 구간을 버려도 bin 번호는 다시 매기지 않는다 → 예측 배열의 k번째와 항상 같은 시각
    rel_t = frame_t - frame_t[0]
    duration = rel_t[-1] + np.median(np.diff(rel_t))
    bin_start = time_bins(duration, cfg["bin_sec"])
    bins = np.arange(len(bin_start))
    starts = np.searchsorted(rel_t, bin_start - 1e-9, side="left")
    centers = frame_t[0] + bin_start + cfg["bin_sec"] / 2

    # 0.01초 격자로 보간 → 이동평균 → 미분. 센서 시작/끝 바깥 구간은 버린다
    t0 = max(frame_t[0], sp_t[0], g_t[0])
    t1 = min(frame_t[-1], sp_t[-1], g_t[-1])
    grid = np.arange(t0, t1, 0.01)
    n_smooth = int(round(cfg["smooth_sec"] / 0.01))
    speed_g = moving_average(np.interp(grid, sp_t, sp), n_smooth)
    accel_g = np.gradient(speed_g, grid)
    yaw_g = moving_average(np.interp(grid, g_t, gyro[:, 2]), n_smooth)

    valid = (centers >= t0) & (centers <= t1)
    bins, bin_start, starts, centers = bins[valid], bin_start[valid], starts[valid], centers[valid]
    speed = np.interp(centers, grid, speed_g)
    accel = np.interp(centers, grid, accel_g)
    # gyro z(+) = 우회전 → 좌회전을 +로 쓰기 위해 부호를 뒤집는다 (조향각과 같은 방향)
    yaw_dps = -np.degrees(np.interp(centers, grid, yaw_g))
    steering = np.interp(centers, st_t, st)

    lon, lat = cfg["longitudinal"], cfg["lateral"]
    long_idx = np.where(accel > lon["accel_mps2"], 2, np.where(accel < lon["decel_mps2"], 0, 1))
    turning = (np.abs(yaw_dps) > lat["yaw_rate_dps"]) & (speed >= lat["min_speed_mps"])
    lat_idx = np.where(turning, np.where(yaw_dps > 0, 0, 2), 1)

    route = seg.split("/")[1]
    return pd.DataFrame({
        "segment": seg.split("/", 1)[1],
        "route": route,
        "split": route_split(route, cfg["test_percent"]),
        "bin": bins,
        "t": bin_start,
        "frame_start": starts,
        "speed": speed.round(3),
        "accel": accel.round(3),
        "yaw_rate_dps": yaw_dps.round(3),
        "steering_deg": steering.round(2),
        "long_label": LONG_NAMES[long_idx],
        "lat_label": LAT_NAMES[lat_idx],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", help="예: Chunk_1 Chunk_2 (기본: zips/ 안의 전부)")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "stage3.yaml").read_text(encoding="utf-8"))
    comma_dir = load_paths()["comma_dir"]
    zip_paths = sorted((comma_dir / "zips").glob("Chunk_*.zip"))
    if args.chunks:
        zip_paths = [p for p in zip_paths if p.stem in args.chunks]
    if not zip_paths:
        sys.exit(f"Chunk zip 없음: {comma_dir / 'zips'}")

    out_dir = comma_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for zpath in zip_paths:
        with zipfile.ZipFile(zpath) as z:
            names = set(z.namelist())
            segs = sorted(n.rsplit("/", 1)[0] for n in names if n.endswith("/video.hevc"))
            frames, failed = [], []
            for seg in tqdm(segs, desc=zpath.stem):
                try:
                    frames.append(segment_labels(z, seg, cfg))
                except (KeyError, ValueError) as e:
                    failed.append((seg, repr(e)))
        df = pd.concat(frames, ignore_index=True)
        out = out_dir / f"{zpath.stem}.csv.gz"
        df.to_csv(out, index=False, encoding="utf-8-sig")

        hours = len(df) * cfg["bin_sec"] / 3600
        print(f"\n{zpath.stem}: 조각 {len(segs) - len(failed)}/{len(segs)}개, 주행 {df['route'].nunique()}회, "
              f"{hours:.2f}시간, 라벨 {len(df):,}개 → {out}")
        print(f"  분할(구간 수): {df['split'].value_counts().to_dict()}")
        print(f"  종방향 %: {(df['long_label'].value_counts(normalize=True) * 100).round(1).to_dict()}")
        print(f"  횡방향 %: {(df['lat_label'].value_counts(normalize=True) * 100).round(1).to_dict()}")
        combo = (df["long_label"] + "·" + df["lat_label"]).value_counts(normalize=True) * 100
        print(f"  9개 조합 %: {combo.round(1).to_dict()}")
        for seg, err in failed:
            print(f"  [실패] {seg}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
