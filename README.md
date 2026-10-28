# 블랙박스 영상 기반 고의교통사고 분석 AI

계획서: `블랙박스_고의사고분석AI_프로젝트계획서_박정식.pdf`

영상 하나를 넣으면 세 가지를 출력한다.

| Stage | 질문 | 폴더 |
|---|---|---|
| 1 | 원본인가, 모니터를 다시 찍은 영상인가 | `src/stage1_rebroadcast/` |
| 2 | 충돌·진입 시점은 언제이고 어떤 상황인가 | `src/stage2_timing/` |
| 3 | 내 차는 0.1초마다 가속/감속/정속 × 좌/우/직진 중 무엇인가 | `src/stage3_egomotion/` |

세 Stage는 `src/common/`의 공통 전처리(프레임 추출 → 차량 검출·추적 → 광류)를 공유한다.

## 장비 역할

| 장비 | 역할 |
|---|---|
| 데이터 PC (GPU 없음, AI Hub 원본 130GB) | zip 인벤토리, 블랙박스 영상 선별, 라벨링 |
| RTX 4060 16GB | 공통 전처리 캐시 생성, Stage 2 학습 |
| RTX 5060 8GB | Stage 1, Stage 3 학습 |
| 코랩 | 캐시 파일을 받아 실험 |
| RunPod L40S | 마지막 60분 오프라인 추론 검증만 |

원본 영상은 PC마다 복사하지 않는다. 선별한 영상만 4060 PC로 한 번 옮기고, 거기서 만든 캐시(`data/cache/`)를 나눠 받는다.

## 설치

```bash
pip install -r requirements.txt
```

학습 PC는 PyTorch를 CUDA 12.8 빌드로 먼저 설치한다 (5060은 Blackwell이라 cu128 이상 필수).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-train.txt
```

PC마다 다른 경로는 `configs/paths.local.yaml`에 적는다 (git 제외). 키는 `configs/paths.yaml`과 같다.

## 1단계: 데이터 인벤토리

zip을 풀지 않고 라벨 JSON만 읽어 영상 목록을 만든다.

```bash
python scripts/build_inventory.py
```

- `data/inventory/videos.csv` — 영상 1건당 1행. 촬영 방식(`filming_way`: bb=블랙박스), 사고 분류·장소, 과실비율, 사고 유형 코드, 어느 zip의 어느 파일인지, 프레임 라벨 수
- `data/inventory/zips.csv` — zip 1개당 1행. 어떤 zip을 풀지 정할 때 사용

Windows 콘솔에서 한글이 깨지면 `python -X utf8 ...`로 실행한다.

### AI Hub 라벨 구조 (확인한 내용)

- **영상 라벨** (`*_영상_*.zip`): 영상 1건당 JSON 1개. 영상 단위 메타데이터만 있고 **충돌·진입 시점은 없다** → Stage 2용 300건은 직접 라벨링 (계획서대로).
- **이미지 라벨** (`*_이미지_*.zip`): 영상별 폴더 안에 프레임별 JSON. `objects[]`에 bbox(x, y, w, h)와 `isObjectA`/`isObjectB`(사고 당사자 차량 표시)가 있다 → 상대 차량 검출·추적 학습과 검증에 바로 쓸 수 있다.
- **주의 — AI Hub의 train/val은 영상 단위가 아니다.** 영상 라벨은 train 17,645건 / val 2,207건으로 나뉘지만, 이미지(프레임) 라벨은 같은 영상의 프레임을 약 8:1로 TL/VL에 나눠 넣었다. 따라서 VL 이미지 라벨로 검증하면 학습에 본 영상이 섞인다. **우리 평가용 20%는 `videos.csv`에서 영상 단위로 직접 떼어낸다** (계획서 4장).
- `video_point_of_view`: 1이 17,071건, 3이 2,781건. `filming_way`: bb(블랙박스) 19,720건, cc(CCTV) 132건 → cc는 제외.
- 영상은 전부 **15fps, 10초(150프레임)이고 사고 시점 앞뒤 5초로 잘려 있다** → 충돌이 대체로 75프레임 근처. Stage 2 모델은 "항상 75프레임" 베이스라인과 꼭 비교한다.

## 2단계: 사고유형 코드표

라벨의 `traffic_accident_type`(0~433) 하나가 사고객체·장소·장소특징·A/B 진행방향·과실비율 조합 하나다.
데이터설명서 PDF의 [별첨 ②] 표를 파싱해 `configs/accident_types.csv`로 만들었다 (434/434, 인벤토리의 zip 분류와 전부 일치).

```bash
pip install pdfplumber
python scripts/parse_accident_types.py "docs/aihub/1-56 교통사고 영상 데이터_데이터설명서(수정)_250328.pdf"
```

코드에서는 `src.common.inventory.load_videos()`로 인벤토리와 코드표를 합쳐 읽는다.
**약 1,670건은 사고유형 코드가 비어 있거나(1,648건) 표에 없는 값(434~446, 19건)**이라 `type_known=False`다.

## 3단계: 평가용 분할과 라벨링 대상

```bash
python scripts/make_splits.py --annotators 이름1 이름2 이름3 이름4 이름5 --extract
```

- `data/splits/video_split.csv` — 영상 이름 해시로 20%를 평가용(test)으로 고정 (15,841 / 4,011). 같은 스크립트를 돌리면 누구나 같은 분할이 나온다
- `data/splits/stage2_labeling.csv` — Stage 2 시점 라벨링 300건. 블랙박스·1인칭·코드 확인된 영상 중, 차대차 끼어들기류 120 / 차대차 기타 60 / 이륜차 50 / 자전거 35 / 보행자 35. 사고유형 225종이 섞이고, 50건은 2명 중복 배정 (1인당 70건)
- `--extract` — 대상 영상 300개(2.1GB)를 `data/raw/stage2_labeling/`에 푼다 (약 1분)

라벨링 기준은 [docs/labeling_guide.md](docs/labeling_guide.md).

## 데이터 다룰 때 규칙

AI Hub 약관상 승인 없는 배포·제3자 제공이 금지되어 있고, 해외 서비스(클라우드, AI 도구)로 보내도 되는지는 확인 중이다.
확인 전까지 원본 영상·프레임·라벨 원문은 외부 서비스에 올리지 않고, 공유는 집계 결과와 코드로만 한다.

## 폴더

```
configs/        경로·학습 설정 (yaml)
src/common/     공통 전처리, 경로 유틸
src/stage1_rebroadcast/
src/stage2_timing/
src/stage3_egomotion/
scripts/        한 번 실행하는 작업 (인벤토리, 선별, 캐시 생성)
notebooks/      탐색·실험
data/inventory/ 인벤토리 CSV (git 제외)
data/raw/       선별해 풀어 둔 영상 (git 제외)
data/cache/     전처리 캐시 (git 제외)
docs/           라벨링 기준표, 회의록
```
