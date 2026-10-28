"""AI Hub 교통사고 영상 데이터의 zip을 풀지 않고 인벤토리 CSV를 만든다.

출력 (configs/paths.yaml 의 inventory_dir):
  videos.csv  - 영상 1건당 1행. 라벨 JSON의 메타데이터 + 어느 zip에 들어 있는지 + 영상 파일 크기
  zips.csv    - zip 파일 1개당 1행. 종류/크기/항목 수 (어떤 zip을 풀지 정할 때 사용)

실행:
  python scripts/build_inventory.py            # 전체 (Training + Validation)
  python scripts/build_inventory.py --split 2.Validation   # 한쪽만 (빠른 확인용)
"""
import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import load_paths  # noqa: E402

SPLITS = ["1.Training", "2.Validation"]

# zip 이름: TL_차대보행자_영상_횡단보도(신호등없음).zip
#   접두어  TL/TS/VL/VS  (T=Training, V=Validation / L=라벨, S=원천)
#   사고 분류(차대차·차대보행자·…) _ 영상|이미지 _ 장소
ZIP_NAME_RE = re.compile(r"^(?P<prefix>[TV][LS])_(?P<category>[^_]+)_(?P<kind>영상|이미지)_(?P<place>.+)\.zip$")

VIDEO_FIELDS = [
    "video_name", "video_date", "filming_way", "video_point_of_view",
    "accident_negligence_rateA", "accident_negligence_rateB",
    "accident_object", "accident_place", "accident_place_feature",
    "vehicle_a_progress_info", "vehicle_b_progress_info",
    "traffic_accident_type", "weather",
]


def parse_zip_name(name: str) -> dict | None:
    m = ZIP_NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["split"] = "train" if d["prefix"][0] == "T" else "val"
    d["content"] = "label" if d["prefix"][1] == "L" else "source"
    return d


def scan_zips(root: Path, splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        for zpath in sorted((root / split).rglob("*.zip")):
            info = parse_zip_name(zpath.name)
            if info is None:
                print(f"[skip] 이름 규칙에 안 맞음: {zpath}", file=sys.stderr)
                continue
            with zipfile.ZipFile(zpath) as z:
                members = [m for m in z.infolist() if not m.is_dir()]
            rows.append({
                "zip_name": zpath.name,
                "zip_path": str(zpath.relative_to(root)),
                "split": info["split"],
                "content": info["content"],
                "kind": info["kind"],
                "category": info["category"],
                "place": info["place"],
                "n_files": len(members),
                "size_mb": round(zpath.stat().st_size / 1e6, 1),
                "_members": members,
                "_path": zpath,
            })
    return rows


def build_video_rows(zips: list[dict]) -> list[dict]:
    """영상 라벨 zip(JSON)을 기준으로 행을 만들고, 원천 zip과 이미지 라벨 zip 정보를 붙인다."""
    videos: dict[str, dict] = {}

    # 1) 영상 라벨 JSON → 메타데이터
    label_zips = [z for z in zips if z["content"] == "label" and z["kind"] == "영상"]
    for z in tqdm(label_zips, desc="영상 라벨"):
        with zipfile.ZipFile(z["_path"]) as zf:
            for m in z["_members"]:
                if not m.filename.endswith(".json"):
                    continue
                d = json.loads(zf.read(m).decode("utf-8-sig"))
                v = d.get("video", {})
                name = v.get("video_name") or Path(m.filename).stem
                row = {k: v.get(k, "") for k in VIDEO_FIELDS}
                row["video_name"] = name
                row["split"] = z["split"]
                row["category"] = z["category"]
                row["place"] = z["place"]
                row["label_zip"] = z["zip_name"]
                row["source_zip"] = ""
                row["video_file"] = ""
                row["video_size_mb"] = ""
                row["n_frame_labels_train"] = 0
                row["n_frame_labels_val"] = 0
                videos[name] = row

    # 2) 원천 영상 zip → 파일 위치·크기
    source_zips = [z for z in zips if z["content"] == "source" and z["kind"] == "영상"]
    for z in tqdm(source_zips, desc="원천 영상"):
        for m in z["_members"]:
            name = Path(m.filename).stem
            row = videos.get(name)
            if row is None:
                continue
            row["source_zip"] = z["zip_name"]
            row["video_file"] = m.filename
            row["video_size_mb"] = round(m.file_size / 1e6, 1)

    # 3) 이미지 라벨 zip → 프레임 단위 bbox 라벨 개수 (영상별 폴더에 들어 있음)
    #    주의: AI Hub는 프레임을 train/val로 나눴다. 같은 영상의 프레임이 TL과 VL 양쪽에 들어 있으므로
    #    영상 단위 홀드아웃이 아니다. 그래서 split별로 따로 센다.
    frame_zips = [z for z in zips if z["content"] == "label" and z["kind"] == "이미지"]
    for z in tqdm(frame_zips, desc="프레임 라벨"):
        col = f"n_frame_labels_{z['split']}"
        for m in z["_members"]:
            if not m.filename.endswith(".json"):
                continue
            name = m.filename.split("/")[0]
            row = videos.get(name)
            if row is None:
                continue
            row[col] += 1

    return sorted(videos.values(), key=lambda r: (r["split"], r["category"], r["video_name"]))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, action="append", help="지정 안 하면 전체")
    args = ap.parse_args()

    paths = load_paths()
    root = paths["aihub_root"]
    out_dir = paths["inventory_dir"]
    if not root.exists():
        sys.exit(f"aihub_root 없음: {root}  (configs/paths.yaml 확인)")

    zips = scan_zips(root, args.split or SPLITS)
    videos = build_video_rows(zips)

    zip_fields = ["zip_name", "zip_path", "split", "content", "kind", "category", "place", "n_files", "size_mb"]
    write_csv(out_dir / "zips.csv", zips, zip_fields)

    video_fields = ["video_name", "split", "category", "place", *VIDEO_FIELDS[1:],
                    "source_zip", "video_file", "video_size_mb", "label_zip",
                    "n_frame_labels_train", "n_frame_labels_val"]
    write_csv(out_dir / "videos.csv", videos, video_fields)

    # 요약
    n_bb = sum(1 for r in videos if r["filming_way"] == "bb")
    n_src = sum(1 for r in videos if r["source_zip"])
    n_frames = sum(1 for r in videos if r["n_frame_labels_train"] or r["n_frame_labels_val"])
    total_mb = sum(r["video_size_mb"] or 0 for r in videos)
    print(f"\nzip {len(zips)}개 → {out_dir / 'zips.csv'}")
    print(f"영상 {len(videos)}건 → {out_dir / 'videos.csv'}")
    print(f"  블랙박스(bb) {n_bb}건 / 원천 영상 있음 {n_src}건 / 프레임 라벨 있음 {n_frames}건")
    print(f"  원천 영상 합계 {total_mb / 1000:.1f} GB")
    by_cat: dict[str, int] = {}
    for r in videos:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
