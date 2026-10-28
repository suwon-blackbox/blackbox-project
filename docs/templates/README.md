# 공유 시트 양식

CSV를 Google Sheets로 가져와(파일 → 가져오기) 두 사람이 같이 쓴다. 첫 줄 예시 행은 지우고 사용한다.

## gpu_schedule.csv — GPU 예약·사용 기록

두 대(5060, 4060)를 누가 언제 쓰는지와 RunPod 비용을 한 표에서 관리한다. 계획서 8장의 "사용 기록표" 역할.

| 열 | 값 |
|---|---|
| `gpu` | `RTX5060` / `RTX4060` / `RunPod-L40S` / `Colab` |
| `person` | `A` / `B` (docs/interfaces.md 역할) |
| `stage` | `common` / `stage1` / `stage2` / `stage3` / `integration` |
| `wandb_run` | W&B 실험 이름. 규칙: `stage<N>-<모델>-<입력>-v<번호>-<사람>` |
| `planned_hours` / `actual_hours` | 예약 시간 / 실제 사용 시간 (주간 회의에서 확인) |
| `runpod_cost_usd` | RunPod 사용 시에만 |

## recapture_log.csv — Stage 1 재촬영 기록

재녹화 영상 1개당 1행. **촬영 조건을 남겨야 "어떤 조건에서 틀리는지"를 분석할 수 있다.**

| 열 | 값 |
|---|---|
| `clip_id` | `R0001`부터. 파일 이름과 같게 |
| `source_video` | 화면에 띄운 원본 영상 이름 |
| `source_split` | 원본의 `split_ours`. **재녹화본은 원본과 같은 분할을 따른다** (같은 원본이 train/test에 섞이면 점수가 부풀려짐) |
| `display_type` | `monitor` / `laptop` / `tv` / `tablet` / `phone` |
| `refresh_hz` | 60, 120, 144 … (모아레·깜빡임 패턴에 영향) |
| `distance_cm`, `angle_deg` | 화면까지 거리, 정면 기준 기울기 (0 = 정면) |
| `room_light` | `dark` / `normal` / `bright` |
| `screen_brightness` | `low` / `mid` / `high` |
| `holding` | `handheld` / `tripod` |
| `orientation` | `landscape` / `portrait` |

조건은 골고루 바꾼다. 같은 조건만 반복하면 모델이 그 조건만 외운다.
