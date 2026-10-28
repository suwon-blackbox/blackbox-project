"""PC 환경 점검. 새 PC에서 처음 한 번, 그리고 문제가 생기면 실행한다.

  python scripts/check_env.py           # 전체
  python scripts/check_env.py --quick   # 영상 디코딩 시험 생략

결과: [OK] 정상 / [WARN] 이 PC 역할에 따라 괜찮을 수 있음 / [FAIL] 고쳐야 함
"""
import argparse
import importlib
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

results: list[tuple[str, str, str]] = []


def report(level: str, item: str, detail: str = "") -> None:
    results.append((level, item, detail))
    print(f"[{level:4}] {item}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    level = "OK" if (3, 10) <= (v.major, v.minor) else "FAIL"
    report(level, f"Python {platform.python_version()}", "3.10 이상 필요" if level == "FAIL" else "")


def check_packages() -> None:
    for name, role in [("numpy", "공통"), ("pandas", "공통"), ("cv2", "공통"), ("yaml", "공통"), ("tqdm", "공통"),
                       ("pdfplumber", "데이터 PC"), ("pytest", "개발")]:
        try:
            mod = importlib.import_module(name)
            report("OK", f"{name} {getattr(mod, '__version__', '')}")
        except ImportError:
            level = "FAIL" if role == "공통" else "WARN"
            report(level, f"{name} 없음", f"{role}용 → pip install -r requirements.txt")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        report("WARN", "PyTorch 없음", "데이터 PC면 정상. 학습 PC는 README의 cu128 설치 명령 실행")
        return
    report("OK", f"PyTorch {torch.__version__} (CUDA 빌드 {torch.version.cuda})")
    if not torch.cuda.is_available():
        level = "WARN" if torch.version.cuda is None else "FAIL"
        report(level, "CUDA 사용 불가", "CPU 전용 빌드 → 학습 PC면 cu128 빌드로 다시 설치" if torch.version.cuda is None
               else "드라이버 확인 (nvidia-smi)")
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = f"sm_{p.major}{p.minor}"
        report("OK", f"GPU {i}: {p.name}", f"VRAM {p.total_memory / 2**30:.1f}GB, {cc}")
        # RTX 50xx(Blackwell, sm_120)는 CUDA 12.8 이상 빌드가 필요
        if p.major >= 12 and (torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8)):
            report("FAIL", f"{p.name}는 CUDA 12.8 이상 PyTorch 필요", "pip install torch --index-url https://download.pytorch.org/whl/cu128")
    try:
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        (x @ x).sum().item()
        report("OK", "GPU FP16 연산 시험")
    except Exception as e:  # noqa: BLE001 — 어떤 오류든 그대로 보여준다
        report("FAIL", "GPU 연산 시험 실패", repr(e))


def check_tools() -> None:
    ffmpeg = shutil.which("ffmpeg")
    report("OK" if ffmpeg else "WARN", "ffmpeg", ffmpeg or "없음 (OpenCV만으로도 읽기는 됨. 영상 변환 작업에만 필요)")
    try:
        import cv2
        report("OK" if cv2.getBuildInformation().count("FFMPEG") else "FAIL", "OpenCV FFMPEG 백엔드")
    except ImportError:
        pass


def check_paths() -> None:
    from src.common.paths import load_paths
    paths = load_paths()
    for key, p in paths.items():
        exists = p.exists()
        level = "OK" if exists else ("WARN" if key in {"aihub_root", "comma_dir", "raw_dir", "cache_dir"} else "OK")
        report(level, f"경로 {key}", f"{p}" + ("" if exists else "  (없음 — 이 PC에 해당 데이터가 없으면 정상)"))
    usage = shutil.disk_usage(paths["cache_dir"].anchor or ".")
    free_gb = usage.free / 1e9
    report("OK" if free_gb > 100 else "WARN", f"디스크 여유 {free_gb:.0f}GB", "" if free_gb > 100 else "캐시용으로 100GB 이상 권장")


def check_decoding() -> None:
    from src.common.paths import load_paths
    from src.common.video import open_comma_segment, open_video
    paths = load_paths()

    mp4s = sorted(paths["raw_dir"].rglob("*.mp4"))[:1]
    if mp4s:
        try:
            v = open_video(mp4s[0])
            v.read(v.n_frames // 2)
            ok = abs(v.fps - 15) < 0.5
            report("OK" if ok else "WARN", f"AI Hub mp4 디코딩 ({v.fps:.1f}fps, {v.n_frames}프레임)",
                   "" if ok else "AI Hub는 15fps여야 함")
        except Exception as e:  # noqa: BLE001
            report("FAIL", "AI Hub mp4 디코딩", repr(e))
    else:
        report("WARN", "AI Hub mp4 시험 생략", "data/raw 에 mp4 없음")

    zips = sorted((paths["comma_dir"] / "zips").glob("Chunk_*.zip"))[:1]
    if zips:
        try:
            with zipfile.ZipFile(zips[0]) as z:
                seg = next(n for n in z.namelist() if n.endswith("/video.hevc")).rsplit("/", 1)[0]
            with tempfile.TemporaryDirectory() as tmp:
                v = open_comma_segment(zips[0], seg, tmp)
                v.read(100)
                ok = abs(v.fps - 20) < 0.1
                report("OK" if ok else "FAIL", f"comma2k19 HEVC 디코딩 ({v.fps:.1f}fps, {v.n_frames}프레임)",
                       "" if ok else "20fps여야 함 — frame_times 처리 확인")
                v.close()
        except Exception as e:  # noqa: BLE001
            report("FAIL", "comma2k19 HEVC 디코딩", repr(e))
    else:
        report("WARN", "comma2k19 시험 생략", "zip 없음")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print(f"== {platform.node()} / {platform.platform()}\n")
    check_python()
    check_packages()
    check_torch()
    check_tools()
    check_paths()
    if not args.quick:
        check_decoding()

    n_fail = sum(1 for lv, _, _ in results if lv == "FAIL")
    n_warn = sum(1 for lv, _, _ in results if lv == "WARN")
    print(f"\n요약: FAIL {n_fail}개, WARN {n_warn}개")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
