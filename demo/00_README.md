# 낙상 감시 모니터 (demo)

병원 간호사실 모니터처럼 CCTV 4대 화면을 띄우고, **우리 최종 모델(R-b)로 실시간 낙상 판정**을 하는 데모입니다.
영상(AI Hub 시험 클립)을 실시간처럼 15fps로 틀면서, 매 프레임 관절 추출 → TCNTE 판정 → 2단계 판정 → 경보까지 실제로 돌립니다.

> **AI Hub 영상은 이 저장소에 없습니다.** 팀 내부용·외부 공개 금지라서 공개 저장소에 올리지 않았습니다. 팀 채널로 받은 `demo_영상_AIHub_팀내부용.zip`을 `monitor\clips\`에 풀어 넣으세요 (3단계). 화면 녹화도 외부에 공개하지 않습니다.

## 1. 필요한 것
| 항목 | 내용 |
|---|---|
| OS | Windows 10/11 (만든 PC: Windows 11) |
| GPU | **NVIDIA GPU 권장.** 만든 PC는 RTX 4060 8GB에서 4대 모두 약 15fps. GPU가 없으면 CPU로 돌아가지만 많이 느림 (시험 안 해 봄) |
| Python | 3.10 이상 (만든 PC: 3.11) |
| 디스크 | 약 40MB (코드·모델 + 영상) |

## 2. 설치 (처음 한 번)
Miniconda를 쓰는 경우 예시입니다. 명령은 **Anaconda Prompt**에서 입력합니다.

```
conda create -n fall python=3.11 -y
conda activate fall
```

**① PyTorch 설치** — GPU가 있으면 CUDA용으로 설치합니다.
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
- GPU가 없으면: `pip install torch`
- 드라이버가 오래돼서 cu126이 안 되면 https://pytorch.org/get-started/locally/ 에서 내 PC에 맞는 명령을 고릅니다.

**② 나머지 설치** — 이 폴더로 이동한 뒤:
```
cd 이 폴더 경로
pip install -r requirements.txt
```

**③ 설치 확인**
```
python -c "import torch, ultralytics, cv2, lap; print(torch.__version__, torch.cuda.is_available())"
```
- 맨 끝이 `True`면 GPU를 씁니다. `False`면 CPU로 돌아갑니다 (느림).

만든 PC의 버전 (참고): torch 2.14.1+cu126, ultralytics 8.4.215, opencv 5.0.0, numpy 2.4.6, lap 0.5.14

## 3. 영상 넣기 (처음 한 번)
- 팀 채널로 받은 `demo_영상_AIHub_팀내부용.zip`을 풀어서, mp4 13개를 `demo\monitor\clips\` 폴더에 바로 넣습니다.
- 파일 목록은 `monitor\clips\README.md`에 있습니다.

## 4. 실행
- **방법 1:** `실행.bat` 더블클릭.
  - conda 환경 이름이 `fall`이면 자동으로 찾습니다.
  - 다른 이름이면 방법 2를 쓰세요.
- **방법 2:** Anaconda Prompt에서
  ```
  conda activate fall
  cd 이 폴더 경로
  python -X utf8 monitor\server.py
  ```

실행 후:
1. 브라우저가 **http://127.0.0.1:8765** 로 열립니다 (안 열리면 주소를 직접 입력).
2. 모델을 불러오는 데 10초쯤 걸리고, 4대가 다 준비되면 **동시에** 재생이 시작됩니다.
3. 끌 때는 검은 실행 창에서 **Ctrl+C**.

## 5. 화면 보는 법
| 영역 | 내용 |
|---|---|
| 카메라 4대 | 뼈대·박스 색이 상태: **초록** 정상 · **노랑** 넘어짐 의심 · **주황** 쓰러짐 확인 · **빨강** 낙상 경보 |
| 경보 | 화면 테두리가 빨갛게 깜빡이고 '낙상 경보' 카드가 뜸 → **확인** 버튼으로 해제 |
| 오른쪽 | 구역별 상태, 이벤트 기록, 판정 설정 |
| 단축키 | **R** 처음부터 다시 (4대 동시) · **F** 전체 화면 · **1~4** 크게 보기 · **Esc** 돌아가기 |
| 경보음 | 오른쪽 위 '경보음 꺼짐' 버튼을 한 번 눌러야 소리가 남 (브라우저 정책) |

## 6. 카메라 구성 (카메라마다 장소 고정, 상황은 이어서 재생)
| 카메라 | 장소 | 재생 순서 (판정 결과) |
|---|---|---|
| CAM 01 | 병실 1 | 앞으로 넘어짐 (경보) → 허리 숙이기 (정상) → 뒤로 넘어짐 (경보) |
| CAM 02 | 복도 1 | 보행기 걷기 (정상) → 뒤로 넘어짐 (경보) → 허리 숙이기 (정상) → 걷기 (정상) |
| CAM 03 | 복도 2 | **보행기 짚고 걷다 주저앉을 뻔 → 버티고 일어섬 (넘어짐 의심까지만, 경보 없음)** → 앞으로 넘어짐 (경보) → 뒤로 넘어짐 (경보) |
| CAM 04 | 생활실 | 걷기 (정상) → 앞으로 넘어짐 (경보) → 뒤로 넘어짐 (경보) |

- 모든 영상은 학습에 안 쓴 배우의 AI Hub 시험 영상입니다.
- CAM 03 첫 장면은 넘어질 뻔한 동작을 '의심'으로 잡지만, 쓰러진 상태가 이어지지 않아 **경보를 내지 않는** 예입니다 (2단계 판정).
- 경보 대기는 데모용 1초입니다 (현장 권장 5초). `monitor\config.json`의 `alarm_s`로 바꿀 수 있습니다.

## 7. 폴더 구성
| 경로 | 내용 |
|---|---|
| `실행.bat` | 더블클릭 실행 |
| `requirements.txt` | 설치 목록 (PyTorch는 2단계 ①로 따로) |
| `monitor\server.py` | 실행 파일. 카메라 4대를 각각 프로세스로 띄우고 웹 화면 제공 |
| `monitor\engine.py` | 카메라 1대의 판정 루프 |
| `monitor\config.json` | 카메라 이름·재생 순서·경보 대기 등 설정 |
| `monitor\web\` | 화면 (html·css·js) |
| `monitor\clips\` | 영상 13개를 넣는 곳 (저장소에는 없음, 따로 받기). AI Hub 원본 4K를 15fps 1280×720으로 줄인 것 |
| `demo_video.py`, `common.py`, `features.py`, `models.py` | 판정 규칙, 특징 55개, 모델 구조 (원본 프로젝트 코드 그대로) |
| `runs\Rb_r2_sel\tcnte_seed0~2\` | 최종 판정 모델 3개 (`best.pt`, `config.json`) |
| `yolo26n-pose.pt` | 관절 추출 모델 |

- 처음 실행하면 `raw`, `labels`, `logs` 같은 빈 폴더가 생깁니다. 원본 코드가 만드는 것이라 무시해도 됩니다.

## 8. 안 될 때
| 증상 | 해결 |
|---|---|
| `클립이 없습니다: [...]` | 3단계: 영상 13개를 `monitor\clips\`에 넣었는지 확인 |
| `포트 8765를 이미 쓰고 있습니다` | 이미 켜 둔 모니터 창을 Ctrl+C로 끄거나, `python -X utf8 monitor\server.py --port 8766` |
| `ModuleNotFoundError` | 2단계 설치를 다시 확인 (`conda activate fall` 했는지) |
| 첫 실행에 `lap` 설치 메시지가 뜸 | ultralytics가 자동 설치하는 것. 인터넷이 연결돼 있으면 그대로 기다리면 됨 |
| 영상이 느림 (fps가 15보다 많이 낮음) | GPU를 안 쓰는 경우 (`torch.cuda.is_available()`이 False). PyTorch를 CUDA용으로 다시 설치 |
| 화면에 "서버를 다시 켜 주세요" | 예전 버전 서버가 켜져 있음. 그 창을 끄고 다시 실행 |
