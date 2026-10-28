"""Stage 2 시점 라벨링 도구 (설치 필요 없음, 영상은 이 PC 밖으로 나가지 않음).

  python scripts/label_stage2.py --annotator 정식
  python scripts/label_stage2.py --annotator 팀원 --redo      # 이미 한 것도 다시 보기

data/splits/stage2_labeling.csv 에서 내 담당 영상만 열고, 두 사람이 같이 하는 30건을 먼저 보여준다.
결과: data/labels/stage2/<annotator>.csv (Enter를 누를 때마다 저장, 껐다 켜면 이어서)

조작 (H로 화면에 표시)
  ← → 또는 A D   1프레임 이동          ↑ ↓ 또는 W S   10프레임 이동
  Space          재생/정지 (15fps)
  C  충돌 프레임 = 지금        E  진입 프레임 = 지금        X  진입 없음
  1 left  2 right  3 front  4 rear  0 none   (진입 방향)
  Y / N / U      피할 공간 yes / no / unknown
  Q              품질 바꾸기 ok → occluded → no_collision → bad_video
  M              메모 (터미널에 입력)
  B              AI Hub 상대 차량 박스 보이기/숨기기
  Enter          저장하고 다음 영상        P  이전 영상        Esc  저장하고 종료

기준은 docs/labeling_guide.md
"""
import argparse
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import load_paths  # noqa: E402
from src.common.video import open_video  # noqa: E402

DISPLAY_WIDTH = 1280
QUALITIES = ["ok", "occluded", "no_collision", "bad_video"]
SIDES = {ord("1"): "left", ord("2"): "right", ord("3"): "front", ord("4"): "rear", ord("0"): "none"}
# cv2.waitKeyEx 화살표 코드 (Windows / Linux GTK)
LEFT, RIGHT, UP, DOWN = {2424832, 65361}, {2555904, 65363}, {2490368, 65362}, {2621440, 65364}
ENTER, ESC, SPACE = {13, 10}, 27, 32


@dataclass
class Label:
    video_name: str
    annotator: str
    collision_frame: int | None = None
    entry_frame: int | None = None
    entry_side: str = ""
    avoidable_space: str = ""
    quality: str = "ok"
    memo: str = ""
    fps: float = 15.0
    labeled_at: str = ""

    @property
    def collision_t(self):
        return None if self.collision_frame is None else round(self.collision_frame / self.fps, 3)

    @property
    def entry_t(self):
        return None if self.entry_frame is None else round(self.entry_frame / self.fps, 3)

    def problems(self) -> list[str]:
        p = []
        if self.collision_frame is None and self.quality not in ("no_collision", "bad_video"):
            p.append("collision(C) missing - or set quality(Q) to no_collision")
        if self.entry_frame is not None and self.collision_frame is not None and self.entry_frame > self.collision_frame:
            p.append("entry is after collision")
        if not self.entry_side:
            p.append("entry side(1-4,0) missing")
        if not self.avoidable_space:
            p.append("avoidable space(Y/N/U) missing")
        return p


@dataclass
class Session:
    """키 입력 → 상태 변경. 화면과 분리해서 --selftest 로 시험할 수 있게 했다."""
    label: Label
    n_frames: int
    frame: int = 0
    playing: bool = False
    show_boxes: bool = True
    show_help: bool = False
    message: str = ""
    confirm_next: bool = field(default=False, repr=False)

    def handle(self, key: int) -> str | None:
        """반환: None(계속) / 'next' / 'prev' / 'quit' / 'memo'"""
        self.message = ""
        k = key & 0xFF if key < 256 else key
        ch = chr(k).lower() if k < 256 else ""
        step = {**{c: -1 for c in LEFT}, **{c: 1 for c in RIGHT}, **{c: 10 for c in UP}, **{c: -10 for c in DOWN}}
        if key in step or ch in "adws" and ch:
            delta = step.get(key, {"a": -1, "d": 1, "w": 10, "s": -10}.get(ch, 0))
            self.frame = int(np.clip(self.frame + delta, 0, self.n_frames - 1))
            self.playing = False
        elif k == SPACE:
            self.playing = not self.playing
        elif ch == "c":
            self.label.collision_frame = self.frame
        elif ch == "e":
            self.label.entry_frame = self.frame
        elif ch == "x":
            self.label.entry_frame = None
            self.label.entry_side = "none"
        elif k in SIDES:
            self.label.entry_side = SIDES[k]
        elif ch in "ynu" and ch:
            self.label.avoidable_space = {"y": "yes", "n": "no", "u": "unknown"}[ch]
        elif ch == "q":
            self.label.quality = QUALITIES[(QUALITIES.index(self.label.quality) + 1) % len(QUALITIES)]
        elif ch == "b":
            self.show_boxes = not self.show_boxes
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "m":
            return "memo"
        elif ch == "p":
            return "prev"
        elif k == ESC:
            return "quit"
        elif k in ENTER:
            problems = self.label.problems()
            if problems and not self.confirm_next:
                self.message = "CHECK: " + "; ".join(problems) + "  (Enter again to save anyway)"
                self.confirm_next = True
                return None
            self.confirm_next = False
            return "next"
        self.confirm_next = False
        return None

    def tick(self) -> None:
        if self.playing:
            if self.frame < self.n_frames - 1:
                self.frame += 1
            else:
                self.playing = False


# ---------- AI Hub 상대 차량 박스 ----------

def build_box_index(queue: pd.DataFrame, aihub_root: Path, cache_path: Path) -> dict[str, dict[int, list]]:
    """영상 → {프레임 번호(0부터): [(bbox, 'A'|'B')]}. AI Hub sequence_frame_number 는 1부터라 1을 뺀다.

    이미지 라벨 zip은 항목이 수십만 개라 열 때마다 느리다 → 담당 영상 전체를 zip별로 한 번에 읽어 캐시한다.
    """
    index: dict[str, dict[int, list]] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        index = {v: {int(f): b for f, b in frames.items()} for v, frames in raw.items()}
    missing = queue[~queue["video_name"].isin(index)]
    if missing.empty:
        return index

    zip_index = {p.name: p for p in aihub_root.rglob("*_이미지_*.zip")}
    print(f"AI Hub 상대 차량 박스 색인 만드는 중 (처음 한 번, {len(missing)}건)…")
    for source_zip, group in missing.groupby("source_zip"):
        wanted = set(group["video_name"])
        for v in wanted:
            index.setdefault(v, {})
        for prefix in ("TL", "VL"):  # 같은 영상의 프레임이 학습/검증 zip에 나뉘어 있다
            path = zip_index.get(prefix + source_zip[2:].replace("_영상_", "_이미지_"))
            if path is None:
                continue
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    video = n.split("/", 1)[0]
                    if video not in wanted or not n.endswith(".json"):
                        continue
                    d = json.loads(z.read(n).decode("utf-8-sig"))
                    idx = int(d["sequence_frame_number"]) - 1
                    for o in d.get("objects", []):
                        who = "A" if o.get("isObjectA") else "B" if o.get("isObjectB") else None
                        if who:
                            index[video].setdefault(idx, []).append((o["bbox"], who))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index), encoding="utf-8")
    return index


# ---------- 화면 ----------

def render(frames: list[np.ndarray], scale: float, s: Session, boxes: dict[int, list], idx_in_list: int,
           total: int) -> np.ndarray:
    img = frames[s.frame].copy()
    h, w = img.shape[:2]
    lb = s.label
    if s.show_boxes:
        for (x, y, bw, bh), who in boxes.get(s.frame, []):
            color = (0, 0, 255) if who == "A" else (255, 128, 0)
            p1 = (int(x * scale), int(y * scale))
            p2 = (int((x + bw) * scale), int((y + bh) * scale))
            cv2.rectangle(img, p1, p2, color, 2)
            cv2.putText(img, f"AIHub {who}", (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def text(line: str, y: int, color=(255, 255, 255)) -> None:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    t = s.frame / lb.fps
    text(f"[{idx_in_list + 1}/{total}] {lb.video_name}", 25)
    text(f"frame {s.frame:3d}/{s.n_frames - 1}  t={t:5.2f}s  {'PLAY' if s.playing else ''}", 50)
    text(f"collision={lb.collision_frame}  entry={lb.entry_frame}  side={lb.entry_side or '-'}  "
         f"avoidable={lb.avoidable_space or '-'}  quality={lb.quality}", 75, (0, 255, 255))
    if lb.memo:
        text("memo: " + lb.memo.encode("ascii", "replace").decode()[:80], 100)
    if s.message:
        text(s.message[:150], h - 60, (0, 128, 255))
    if s.show_help:
        for i, line in enumerate([
            "<- -> / A D : 1 frame     Up Down / W S : 10 frames     Space : play",
            "C collision   E entry   X no entry   1 left 2 right 3 front 4 rear 0 none",
            "Y/N/U avoidable   Q quality   M memo   B AIHub boxes   H help",
            "Enter save+next   P prev   Esc save+quit",
        ]):
            text(line, 130 + i * 25, (200, 255, 200))
    else:
        text("H: help", 130, (200, 255, 200))

    # 아래 시간 막대: 회색=AI Hub 박스 있는 프레임, 빨강=충돌, 초록=진입, 흰색=현재
    bar_y = h - 20
    cv2.rectangle(img, (0, bar_y - 8), (w, h), (40, 40, 40), -1)
    xs = lambda f: int(f / max(1, s.n_frames - 1) * (w - 1))  # noqa: E731
    for f in boxes:
        cv2.line(img, (xs(f), bar_y), (xs(f), bar_y + 6), (120, 120, 120), 1)
    if lb.entry_frame is not None:
        cv2.line(img, (xs(lb.entry_frame), bar_y - 8), (xs(lb.entry_frame), h), (0, 200, 0), 3)
    if lb.collision_frame is not None:
        cv2.line(img, (xs(lb.collision_frame), bar_y - 8), (xs(lb.collision_frame), h), (0, 0, 255), 3)
    cv2.line(img, (xs(s.frame), bar_y - 8), (xs(s.frame), h), (255, 255, 255), 2)
    return img


# ---------- 저장 ----------

def load_done(path: Path) -> dict[str, Label]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"memo": str, "entry_side": str, "avoidable_space": str})
    out = {}
    for r in df.to_dict("records"):
        out[r["video_name"]] = Label(
            video_name=r["video_name"], annotator=r["annotator"],
            collision_frame=None if pd.isna(r["collision_frame"]) else int(r["collision_frame"]),
            entry_frame=None if pd.isna(r["entry_frame"]) else int(r["entry_frame"]),
            entry_side="" if pd.isna(r["entry_side"]) else r["entry_side"],
            avoidable_space="" if pd.isna(r["avoidable_space"]) else r["avoidable_space"],
            quality=r["quality"], memo="" if pd.isna(r["memo"]) else r["memo"],
            fps=float(r["fps"]), labeled_at=r["labeled_at"])
    return out


def save_all(path: Path, labels: dict[str, Label]) -> None:
    rows = [{**asdict(lb), "collision_t": lb.collision_t, "entry_t": lb.entry_t} for lb in labels.values()]
    cols = ["video_name", "annotator", "collision_frame", "collision_t", "entry_frame", "entry_t",
            "entry_side", "avoidable_space", "quality", "memo", "fps", "labeled_at"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=cols).to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)  # 저장 도중 꺼져도 파일이 깨지지 않게


# ---------- 실행 ----------

def my_queue(annotator: str, splits_dir: Path) -> pd.DataFrame:
    lab = pd.read_csv(splits_dir / "stage2_labeling.csv", encoding="utf-8-sig")
    mine = lab[(lab["annotator_1"] == annotator) | (lab["annotator_2"] == annotator)].copy()
    if mine.empty:
        names = sorted(set(lab["annotator_1"]) | set(lab["annotator_2"].dropna()))
        sys.exit(f"'{annotator}' 담당 영상 없음. 가능한 이름: {names}")
    mine["shared"] = mine["annotator_2"].notna()
    return mine.sort_values("shared", ascending=False, kind="stable").reset_index(drop=True)


def load_frames(path: Path) -> tuple[list[np.ndarray], float, float, int]:
    v = open_video(path)
    scale = DISPLAY_WIDTH / v.width
    size = (DISPLAY_WIDTH, int(round(v.height * scale)))
    frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for _, _, f in v.iter_frames()]
    v.close()
    return frames, scale, v.fps, len(frames)


def run(annotator: str, redo: bool) -> None:
    paths = load_paths()
    queue = my_queue(annotator, paths["splits_dir"])
    out_path = paths["splits_dir"].parent / "labels" / "stage2" / f"{annotator}.csv"
    done = load_done(out_path)
    box_index = build_box_index(queue, paths["aihub_root"], out_path.parent / "aihub_boxes.json")
    video_dir = paths["raw_dir"] / "stage2_labeling"

    todo = list(range(len(queue))) if redo else [i for i in range(len(queue)) if queue.loc[i, "video_name"] not in done]
    print(f"{annotator}: 담당 {len(queue)}건, 완료 {len(done)}건, 이번에 볼 영상 {len(todo)}건 → {out_path}")
    if not todo:
        return
    pos = 0
    cv2.namedWindow("stage2", cv2.WINDOW_AUTOSIZE)
    while 0 <= pos < len(todo):
        row = queue.loc[todo[pos]]
        vpath = video_dir / Path(row["video_file"]).name
        if not vpath.exists():
            sys.exit(f"영상 없음: {vpath}  → python scripts/make_splits.py --extract")
        frames, scale, fps, n = load_frames(vpath)
        boxes = box_index.get(row["video_name"], {})
        label = done.get(row["video_name"]) or Label(row["video_name"], annotator, fps=fps)
        s = Session(label=label, n_frames=n)
        while True:
            cv2.imshow("stage2", render(frames, scale, s, boxes, todo[pos], len(queue)))
            key = cv2.waitKeyEx(int(1000 / fps) if s.playing else 0)
            if key == -1:
                s.tick()
                continue
            action = s.handle(key)
            if action == "memo":
                s.label.memo = input(f"[{row['video_name']}] 메모 입력 후 Enter: ").strip()
            elif action in ("next", "prev", "quit"):
                if action != "prev" or s.label.collision_frame is not None or s.label.quality != "ok":
                    s.label.labeled_at = datetime.now().isoformat(timespec="seconds")
                    done[row["video_name"]] = s.label
                    save_all(out_path, done)
                pos += {"next": 1, "prev": -1, "quit": len(todo)}[action]
                pos = max(pos, 0) if action == "prev" else pos
                break
    cv2.destroyAllWindows()
    print(f"저장: {out_path}  (완료 {len(done)}/{len(queue)}건)")


def selftest() -> None:
    """화면 없이 키 처리와 저장을 시험한다."""
    lb = Label("v", "tester")
    s = Session(label=lb, n_frames=150)
    for key in [ord("d")] * 5 + [list(UP)[0]] * 7:        # 5 + 70 = 75
        s.handle(key)
    assert s.frame == 75
    s.handle(ord("c"))
    for key in [list(LEFT)[0]] * 25:
        s.handle(key)
    s.handle(ord("e")); s.handle(ord("2")); s.handle(ord("n"))  # noqa: E702
    assert (lb.collision_frame, lb.entry_frame, lb.entry_side, lb.avoidable_space) == (75, 50, "right", "no")
    assert lb.collision_t == 5.0 and lb.problems() == []
    s2 = Session(label=Label("v2", "tester"), n_frames=150)
    assert s2.handle(13) is None and "collision" in s2.message   # 빠진 항목이 있으면 한 번 막는다
    assert s2.handle(13) == "next"                                # 한 번 더 누르면 저장
    for _ in range(3):
        s2.handle(ord("q"))
    assert s2.label.quality == "bad_video"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.csv"
        save_all(p, {"v": lb})
        back = load_done(p)["v"]
        assert (back.collision_frame, back.entry_side, back.quality) == (75, "right", "ok")
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.annotator:
        run(args.annotator, args.redo)
    else:
        ap.error("--annotator 이름 필요 (예: --annotator 정식)")


if __name__ == "__main__":
    main()
