# 낙상 감지 전이학습 (최종 모델 R-b + 학습 코드)

새 데이터셋으로 **최종 모델(R-b)에서 이어서 학습**하는 코드입니다. 영상과 정답만 준비하면 바로 학습할 수 있습니다.

## 모델이 하는 일
1. CCTV 영상을 15fps로 읽습니다.
2. YOLO26n-pose + ByteTrack으로 사람의 관절 17개를 뽑습니다.
3. 관절에서 특징 55개를 계산합니다.
4. 최근 2초(30프레임)를 TCNTE에 넣어 **매 프레임 0 정상 / 1 넘어지는 중 / 2 쓰러져 있음**을 고릅니다.
5. 2단계 판정(넘어짐 의심 → 쓰러짐 지속)으로 경보를 냅니다.

최종 모델의 설정, 결과, 실험 기록은 `참고_인수인계_최종모델.md`에 있습니다.

## 0. 설치 (처음 한 번)
Python **3.11**을 권장합니다. 3.13 이상에서는 패키지가 안 깔릴 수 있습니다. 명령은 Anaconda Prompt에서 입력합니다.
```
conda create -n fall python=3.11 -y
conda activate fall
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
- GPU가 없으면 torch는 `pip install torch`로 설치합니다. 관절 추출과 학습이 많이 느립니다.
- 아래 명령은 모두 **이 폴더에서** 실행합니다.
- 결과 폴더(`pose`, `pose_sel`, `seq`, `runs`, `logs`)는 자동으로 생깁니다.

## 1. 영상 넣기
`raw\mydata\` 폴더를 만들고 영상을 넣습니다.
- `mydata`는 원하는 이름으로 바꿔도 됩니다. 단, 아래 명령과 CSV에서 모두 같은 이름을 써야 합니다.

## 2. 정답표 2개 만들기 (`labels\` 폴더)
예시 파일 `labels\videos_mydata_예시.csv`와 `labels\segments_mydata_예시.csv`를 복사한 뒤, `_예시`를 뗀 이름으로 저장하고 내용을 채웁니다.

**`labels\videos_mydata.csv`** — 영상 목록 (한 줄에 영상 하나)

| 열 | 내용 |
|---|---|
| dataset | 데이터셋 이름 (예: `mydata`). 모든 줄에 같은 값 |
| path | 영상을 구별하는 이름. segments의 path와 같아야 함 |
| split | `train` / `val` / `test`. **같은 사람이 여러 분할에 들어가지 않게** 나눔 |
| subject | 사람 번호 (기록용) |
| light | `day` / `night` (기록용) |
| vid | 결과 파일 이름. 보통 path와 같게 쓰고 `/`는 쓰지 않음 |
| video_file | 이 폴더 기준 영상 경로 (예: `raw/mydata/a.mp4`) |

**`labels\segments_mydata.csv`** — 시간 구간별 정답

| 열 | 내용 |
|---|---|
| dataset, path, split | 위 표와 같은 값 |
| label | **0 정상 / 1 넘어지는 중 / 2 쓰러져 있음**. 그냥 누워 있는 상태처럼 학습에서 빼고 싶은 구간은 6 |
| start, end | 구간 시작·끝 (초 단위, 영상 시작 = 0) |

- 구간에 없는 시간은 학습과 채점에서 빠집니다.
- '1 넘어지는 중'은 균형을 잃기 시작해서 바닥에 닿을 때까지입니다. 보통 0.5~1.5초입니다.
- 스스로 눕는 동작(침대에 눕기 등)은 0입니다. 이런 영상을 넣으면 오경보가 줄어듭니다.

## 3. 학습 (순서대로)
```
python 03_extract_pose.py --dataset mydata --videos-csv videos_mydata.csv
python 04_select_and_label.py --tag mydata --datasets mydata --single mydata
python 05_build_sequences.py
python 06_train.py --model tcnte --seed 0 --exp FT_mydata --init runs/Rb_r2_sel/tcnte_seed0/best.pt --lr 1e-4 --hn-weight 3 --max-epochs 30
```

**03 관절 추출**
- 끊겨도 다시 실행하면 이어서 합니다.

**04 주인공 고르기 + 정답 붙이기**
- 영상마다 주인공 1명을 고르고, 프레임마다 정답을 붙입니다.
- `--single mydata`는 영상에 사람이 1명일 때 씁니다. 여러 명이 나오면 빼세요.
- 화면의 **[확인 1] 쓰러진 사람 관절 검출률이 70% 이상**이어야 합니다.
- `logs\label_check_mydata\*.png`를 열어 정답이 바뀌는 시점이 장면과 맞는지 확인합니다.

**05 특징 계산**
- `seq\train.pkl`, `val.pkl`, `test.pkl`을 만듭니다.
- 출력의 '합성 n개'는 원래 프로젝트에서 쓰던 문구입니다. 여기서는 내 데이터의 학습 조각 수입니다.

**06 학습**
- R-b의 시드 0 모델에서 출발합니다.
- val macro-F1이 가장 높은 에폭이 `runs\FT_mydata\tcnte_seed0\best.pt`로 저장됩니다.
- 최종 모델은 시드 3개의 앙상블입니다. 똑같이 하려면 seed와 출발 모델을 1, 2로 바꿔 두 번 더 돌립니다.
  - 예: `--seed 1 --init runs/Rb_r2_sel/tcnte_seed1/best.pt`

**옵션 고르기**

| 상황 | 추천 |
|---|---|
| 데이터가 적음 (영상 수십 개) | `--lr 1e-4 --max-epochs 10~30`. 학습률이 크면 원래 성능을 빨리 잊음 |
| 데이터가 많음 (수백 개 이상) | `--lr 1e-4`부터 시작하고, 점수가 안 오르면 `--lr 3e-4` |
| GPU 메모리 부족 | `--bs 32` |
| 처음부터 학습한 것과 비교 | `--init`을 빼고 `--lr 1e-3` |

- `--hn-weight 3`은 '스스로 눕는 동작'(원래 데이터의 label 5) 구간의 가중치입니다. 내 데이터에 5가 없으면 아무 영향이 없습니다.

## 4. 새 모델 확인
```
python 07_evaluate.py --runs runs/FT_mydata/tcnte_seed0 --sets test --smooth 3 --objective spec95 --report-val --labels-tag mydata
python demo_video.py --video 시험영상.mp4 --models runs/FT_mydata/tcnte_seed0/best.pt --alarm-s 5 --out 결과
```

**07 평가**
- val로 판정값을 고른 뒤 test 결과를 냅니다. 결과는 `results.csv`에 저장됩니다.
- **판정값은 val로만 고르고, test는 마지막에 한 번만 봅니다.**
- `--labels-tag`는 내 정답표를 읽게 하려고 이 꾸러미에서 새로 넣은 옵션입니다.

**demo_video 영상 판정**
- 영상에 뼈대와 상태를 그려 `logs\demo\결과_demo.mp4`로 저장합니다.
- 앙상블로 판정하려면 `--models` 뒤에 best.pt 3개를 적습니다.

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `03_extract_pose.py` ~ `07_evaluate.py` | 관절 추출 → 사람 선택·정답 → 특징 계산 → 학습 → 평가 |
| `common.py`, `features.py`, `models.py`, `train_utils.py` | 경로·라벨 변환, 특징 55개, 모델 구조(TCNTE), 학습 보조 |
| `demo_video.py` | 영상 한 개에 판정을 그려 저장 (실시간 판정 규칙) |
| `runs\Rb_r2_sel\tcnte_seed0~2\` | **최종 모델 R-b** (`best.pt` + `config.json`) |
| `yolo26n-pose.pt` | 관절 추출 모델 |
| `labels\` | 내 데이터 정답표 예시 |
| `참고_인수인계_최종모델.md` | 최종 모델 설정, test 결과, 실험 기록, 알려진 한계 |

## 확인한 것
이 폴더 그대로 전체 흐름을 끝까지 돌려 봤습니다 (RTX 4060).
- 03 → 04 → 05 → 06(R-b에서 이어서 학습) → 07 → demo_video
- 시험 데이터는 AI Hub 영상 7개를 가짜 '내 데이터셋'으로 쓴 것입니다.
