# Stage 간 연결 규격 (초안 v0)

Stage별로 따로 개발하더라도 **이 문서의 형식만 지키면 8주차 통합 때 그대로 붙는다.**
형식을 바꿔야 하면 PR로 이 문서를 먼저 고치고, 두 사람이 모두 확인한 뒤 코드를 바꾼다.

## 역할 (2명 기준)

GPU가 있는 곳과 하는 일을 맞췄다. 아래 표의 "담당" 열은 이 역할 이름을 쓴다.

| 역할 | 사람 | 장비 | 맡는 것 |
|---|---|---|---|
| **A** | 박정식 | 데이터 PC + RTX 5060 8GB | 일정·데이터 관리, 최종 통합·60분 최적화, **Stage 1**, **Stage 3** |
| **B** | 팀원 | RTX 4060 16GB | **공통 전처리 캐시**(검출·추적·광류), **Stage 2** |

- Stage 2가 가장 어렵고 메모리를 많이 쓰므로 16GB 쪽에 둔다. 공통 캐시도 B PC에서 만들어 A에게 넘긴다
- Stage 1은 재촬영 데이터가 모이기 전까지 할 게 적으므로, A가 Stage 3를 먼저 진행한다

```
영상 ──▶ 공통 전처리 (B) ──▶ data/cache/<source>/<video_id>/
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
     Stage 1 (A, 원본 영상 직접)   Stage 2 (B, tracks, lanes)   Stage 3 (A, flow)
              │                           │                           │
              └───────────────▶ 최종 출력 JSON (1영상 = 1파일) ◀───────┘
```

## 0. 공통 약속

| 항목 | 규칙 | 이유 |
|---|---|---|
| **시간 기준** | 모든 시점은 **초 단위 `t`** (영상 시작 = 0.0). `t = frame / fps` | AI Hub 15fps, comma2k19 20fps, 실제 블랙박스는 제각각 → 프레임 번호로는 비교 불가 |
| 프레임 번호 | 0부터. 원본 영상 기준 (추출 간격과 무관) | AI Hub `sequence_frame_number`와 같음 |
| 좌표 | 원본 해상도 픽셀, bbox는 **`[x, y, w, h]` (좌상단 기준)** | AI Hub 라벨과 같은 형식 |
| 좌우 | **내 차(블랙박스 차량) 기준**. 화면 왼쪽 = left | 1인칭 영상 기준 |
| 거동 구간 | 0.1초. 구간 `k`는 `[0.1k, 0.1k+0.1)` 이고 대표 시각은 시작값 `t = 0.1k` | Stage 3 정답 간격 |
| 결측 | 값이 없으면 `null` (JSON) / 빈칸 (CSV). 0이나 -1로 채우지 않는다 | 0이 실제 값일 수 있음 |
| video_id | 확장자 뺀 파일 이름. comma2k19는 `route|segment` → `route_segment` | 폴더 이름에 `|` 사용 불가 |

## 1. 공통 전처리 캐시

폴더: `data/cache/<source>/<video_id>/` (`source` = `aihub`, `comma2k19`, `recapture`)
B의 4060 PC에서 한 번 만들고 A는 이 폴더만 받는다. 원본 영상은 주고받지 않는다.

### 1-1. `meta.json` — 필수

```json
{
  "cache_version": 0,
  "video_id": "sample_001",
  "source": "aihub",
  "split_ours": "train",
  "fps": 15.0,
  "n_frames": 150,
  "duration_sec": 10.0,
  "width": 1920,
  "height": 1080,
  "created_by": "preprocess.py@<git commit>"
}
```

- `split_ours`: AI Hub는 `data/splits/video_split.csv`, comma2k19는 라벨 CSV의 `split` 값을 그대로 쓴다
- 형식을 바꾸면 `cache_version`을 올린다. 읽는 쪽은 버전이 다르면 에러를 낸다

### 1-2. `tracks.csv` — 차량 검출·추적 (Stage 2가 사용)

원본 fps 그대로 모든 프레임에서 검출한다 (ByteTrack은 프레임을 건너뛰면 ID가 잘 끊긴다).

| 컬럼 | 형식 | 설명 |
|---|---|---|
| `frame` | int | 원본 프레임 번호 |
| `t` | float | 초 |
| `track_id` | int | ByteTrack ID. 영상 안에서만 고유 |
| `cls` | str | `vehicle` / `two_wheeler` / `bicycle` / `pedestrian` (AI Hub 클래스명에 맞춤) |
| `x`, `y`, `w`, `h` | float | bbox (원본 픽셀) |
| `score` | float | 검출 신뢰도 0~1 |

### 1-3. `flow.npz` — 화면 움직임 (Stage 3가 사용)

광류는 연속 두 프레임 사이에서 계산한 뒤 **초당 값으로 나누고 0.1초 구간으로 다시 모은다.**
그래야 15fps와 20fps 영상이 같은 입력이 된다.

| 키 | shape | 설명 |
|---|---|---|
| `t` | `(K,)` | 구간 시작 시각 0.0, 0.1, … |
| `grid` | `(K, 4, 4, 2)` | 화면을 4×4 칸으로 나눈 칸별 평균 이동 (dx, dy). **단위: 화면 너비 / 초** (해상도와 무관) |
| `divergence` | `(K,)` | 화면 확장량 (+ 앞으로 다가감) → 가속·감속 단서 |
| `curl` | `(K,)` | 화면 회전량 |
| `mean_dx` | `(K,)` | 전체 평균 가로 이동 → 좌·우 회전 단서 |
| `valid` | `(K,)` bool | 계산 실패·장면 전환 구간은 False |

- 계산 해상도는 가로 480px로 줄여서 한다 (속도). 단위가 화면 너비 기준이라 결과는 같다
- 차량 bbox 영역은 광류 통계에서 뺀다 (다른 차의 움직임이 내 차 움직임으로 보이지 않게)

### 1-4. `lanes.csv` — 차선 (Stage 2가 사용, 형식 미정)

B가 차선 모델을 정한 뒤 이 절을 채운다. 최소한 `frame, t, lane_id, side(left/right)`와 점 좌표는 들어가야 한다.

### 1-5. Stage 1은 캐시를 쓰지 않는다

**재녹화 흔적(모아레)은 고주파 무늬라 해상도를 줄이거나 JPEG로 저장하면 사라진다.**
Stage 1은 원본 영상에서 직접 프레임을 읽고, 저장이 필요하면 원본 해상도 PNG로만 저장한다.

## 2. 최종 출력 JSON

1영상 = 1파일, 파일 이름은 `<video_id>.json`. 전체 예시: [examples/output_example.json](examples/output_example.json)

| 필드 | 형식 | 담당 | 설명 |
|---|---|---|---|
| `schema_version` | int | A | 이 문서 버전 |
| `video_id`, `fps`, `duration_sec` | | A | `meta.json`과 같음 |
| `stage1.label` | `original` / `rebroadcast` | A | |
| `stage1.prob_rebroadcast` | float 0~1 | A | 프레임별 확률 평균 |
| `stage2.collision` | `{t, frame, confidence}` 또는 `null` | B | 첫 접촉 순간. 충돌 없으면 `null` |
| `stage2.entry` | `{t, frame, confidence}` 또는 `null` | B | 상대가 내 차로에 들어오기 시작한 순간 |
| `stage2.entry_side` | `left` / `right` / `front` / `rear` / `none` | B | |
| `stage2.avoidable_space` | `yes` / `no` / `unknown` | B | 진입 순간 피할 빈 공간 |
| `stage2.opponent` | `{track_id, bbox_at_collision}` 또는 `null` | B | `tracks.csv`의 ID |
| `stage3.bin_sec` | `0.1` | A | |
| `stage3.long` | str 배열 (길이 K) | A | `가속` / `감속` / `정속` |
| `stage3.lat` | str 배열 (길이 K) | A | `좌` / `우` / `직진` |
| `stage3.p_long`, `stage3.p_lat` | `(K, 3)` 배열 | A | 클래스 순서는 위 나열 순서. 기준값 재조정용 |
| `runtime_sec` | `{preprocess, stage1, stage2, stage3}` | A | 60분 제한 관리용 |

- Stage 3은 행 목록 대신 **배열**로 낸다 (10초 영상이면 길이 100). 파일이 작고 읽기 쉽다
- 대회 제출 양식이 확인되면 이 JSON → 제출 양식 변환기를 따로 만든다. 이 형식은 바꾸지 않는다

## 3. 평가 규칙

- 모든 Stage는 `split_ours == "test"`인 영상만으로 평가하고, test 영상은 학습·검증·기준값 조정에 쓰지 않는다
- 하이퍼파라미터·기준값은 train 안에서 다시 떼어낸 검증 세트로 조정한다
- 베이스라인과 함께 보고한다
  - Stage 1: 항상 다수 클래스
  - Stage 2: **항상 75프레임(5초)** — AI Hub 클립이 사고 시점 앞뒤 5초로 잘려 있어서 강한 베이스라인이다
  - Stage 3: 항상 `정속·직진`

## 결정할 것 (1주차 회의)

- [ ] 검출 모델 (YOLO 크기)과 `cls` 목록 확정
- [ ] 차선 모델과 `lanes.csv` 형식 (Stage 2)
- [ ] 광류 방식: OpenCV Farneback vs RAFT-small (속도 측정 후)
- [ ] Stage 2 평가 허용 오차 (±3프레임 = 0.2초 제안)
