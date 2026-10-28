# Stage 2 시점 라벨링 기준표 (초안)

대상 목록: `data/splits/stage2_labeling.csv` (200건, `annotator_1` / `annotator_2` 열이 담당자)
영상 위치: `data/raw/stage2_labeling/` (`python scripts/make_splits.py --extract`로 생성)

## 영상 특성 (확인 결과)

- 전부 **15fps, 약 10초(150프레임)**. 해상도는 대부분 1920×1080
- AI Hub가 **사고 시점 기준 앞뒤 5초**로 잘랐다 (데이터설명서 4쪽) → 충돌은 대체로 **5초(75프레임) 근처**
  - 라벨러는 이 사실에 기대지 말고 화면을 보고 찍는다
  - 모델 평가 때 "항상 75프레임이라고 답하는" 베이스라인과 반드시 비교한다

## 라벨 항목

프레임 번호는 0부터 센다 (0 ~ 149). **AI Hub 프레임 라벨의 `sequence_frame_number`는 1부터(1 ~ 150)라서 1을 빼야 같은 프레임이다.** 라벨링 도구는 이미 빼서 보여준다.

| 항목 | 값 | 정의 |
|---|---|---|
| `collision_frame` | 정수 | 내 차(블랙박스 차량)와 상대가 **처음 닿는** 프레임. 닿는 순간이 안 보이면 화면이 흔들리거나 상대가 멈칫하는 첫 프레임 |
| `entry_frame` | 정수 / 빈칸 | 상대가 **내 차로 안으로 들어오기 시작한** 프레임 = 상대 바퀴(또는 차체 끝)가 차선을 처음 넘는 프레임. 끼어들기·진로변경이 아니면 빈칸 |
| `entry_side` | `left` / `right` / `front` / `rear` / `none` | 상대가 들어온 방향 (내 차 기준) |
| `avoidable_space` | `yes` / `no` / `unknown` | 진입 시점에 내 차가 피할 수 있는 빈 차로·갓길이 보였는가 |
| `opponent_box` | (자동) | 따로 표시하지 않는다. 1인칭에서 상대는 사고객체 A라서 AI Hub의 `isObjectA` 박스(도구에서 빨간 박스)를 쓴다 |
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

설치가 필요 없는 자체 도구를 쓴다 (CVAT는 Docker가 필요해서 대신 만듦). 영상은 PC 밖으로 나가지 않는다.

```bash
python scripts/label_stage2.py --annotator 정식      # 또는 --annotator 팀원
```

- 내 담당 영상만 열리고, **두 사람이 같이 하는 30건이 먼저** 나온다 (먼저 10건씩 하고 비교)
- 처음 실행할 때 AI Hub 상대 차량 박스 색인을 만든다 (약 1분, 한 번만)
- 빨간 박스 = AI Hub 사고객체 A(상대). 아래 막대의 회색 눈금 = 박스가 있는 프레임, 빨강 = 충돌, 초록 = 진입
- Enter를 누를 때마다 `data/labels/stage2/<이름>.csv`에 저장된다. 껐다 켜면 이어서 한다
- 빠진 항목이 있으면 Enter를 한 번 막고 알려준다 (한 번 더 누르면 그대로 저장)

| 키 | 동작 |
|---|---|
| ← → / A D | 1프레임 이동 |
| ↑ ↓ / W S | 10프레임 이동 |
| Space | 재생·정지 |
| C / E / X | 충돌 프레임 / 진입 프레임 / 진입 없음 |
| 1 2 3 4 0 | 진입 방향 left / right / front / rear / none |
| Y N U | 피할 공간 yes / no / unknown |
| Q | 품질 ok → occluded → no_collision → bad_video |
| M | 메모 (터미널에 입력) |
| B / H | AI Hub 박스 보이기 / 도움말 |
| Enter / P / Esc | 저장 후 다음 / 이전 / 저장 후 종료 |
