# Stage 2 시점 라벨링 기준표 (초안)

대상 목록: `data/splits/stage2_labeling.csv` (200건, `annotator_1` / `annotator_2` 열이 담당자)
영상 위치: `data/raw/stage2_labeling/` (`python scripts/make_splits.py --extract`로 생성)

## 영상 특성 (확인 결과)

- 전부 **15fps, 약 10초(150프레임)**. 해상도는 대부분 1920×1080
- AI Hub가 **사고 시점 기준 앞뒤 5초**로 잘랐다 (데이터설명서 4쪽) → 충돌은 대체로 **5초(75프레임) 근처**
  - 라벨러는 이 사실에 기대지 말고 화면을 보고 찍는다
  - 모델 평가 때 "항상 75프레임이라고 답하는" 베이스라인과 반드시 비교한다

## 라벨 항목

프레임 번호는 0부터 센다 (0 ~ 149). AI Hub 프레임 라벨의 `sequence_frame_number`와 같다.

| 항목 | 값 | 정의 |
|---|---|---|
| `collision_frame` | 정수 | 내 차(블랙박스 차량)와 상대가 **처음 닿는** 프레임. 닿는 순간이 안 보이면 화면이 흔들리거나 상대가 멈칫하는 첫 프레임 |
| `entry_frame` | 정수 / 빈칸 | 상대가 **내 차로 안으로 들어오기 시작한** 프레임 = 상대 바퀴(또는 차체 끝)가 차선을 처음 넘는 프레임. 끼어들기·진로변경이 아니면 빈칸 |
| `entry_side` | `left` / `right` / `front` / `rear` / `none` | 상대가 들어온 방향 (내 차 기준) |
| `avoidable_space` | `yes` / `no` / `unknown` | 진입 시점에 내 차가 피할 수 있는 빈 차로·갓길이 보였는가 |
| `opponent_box` | bbox | `collision_frame`에서의 상대 bbox. AI Hub 프레임 라벨의 `isObjectA`/`isObjectB` 박스가 있으면 그것을 확인만 한다 |
| `quality` | `ok` / `occluded` / `no_collision` / `bad_video` | 판단이 어려운 영상 표시 |
| `memo` | 글 | 애매했던 이유 |

## 판단 규칙

1. 충돌이 화면 밖에서 나면 충격(화면 흔들림)이 시작된 프레임을 `collision_frame`으로 하고 `quality=occluded`
2. 충돌이 없는 영상(급정거로 피함 등)은 `collision_frame` 빈칸, `quality=no_collision`
3. 차선이 안 보이면 `entry_frame`은 상대 차가 내 차 진행 방향 정면으로 들어오기 시작한 프레임으로 한다
4. 1인칭 영상만 대상이다. 설명서 기준 **1인칭에서는 내 차가 사고객체 B**다

## 일치도 확인 (중복 30건)

- `annotator_2`가 있는 30건은 두 명이 따로 라벨링한다
- 먼저 10건씩 해 보고 프레임 차이를 비교한다. 목표: `collision_frame` 차이 ±3프레임(0.2초) 이내 90% 이상
- 기준이 안 맞는 항목은 이 문서에 규칙을 추가한 뒤 나머지를 진행한다

## 도구

CVAT(무료, 로컬 설치 가능)의 비디오 태스크에서 프레임 태그로 표시한 뒤 CSV로 내보낸다.
이 PC에 CVAT를 띄우면 영상을 외부로 올리지 않아도 된다 (AI Hub 약관 확인 전 권장).
