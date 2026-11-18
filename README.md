# COMPUTER_VISION-10-
컴퓨터 비전 팀 프로젝트: CCTV 기반 실시간 낙상 감지 (제안서 최종 v3 기준, 학습팀 파이프라인)

```
영상 -> YOLO26n-pose + ByteTrack -> 피험자 선택 -> 특징 55차원 -> 분류기(메인 TCNTE) -> 프레임마다 3클래스 확률
```

클래스: 정상 / 넘어지는 중 / 쓰러짐. 학습하는 모델은 분류기 하나이고 포즈 모델은 사전학습 그대로 쓴다.

| 파일 | 제안서 | 역할 |
|---|---|---|
| `download_data.py` | 5장 | omnifall로 영상·라벨 받기, `split.json`(공식 분할), 밤 영상 목록 |
| `extract_pose.py` | 4.3 ① | 15fps 관절 추출, 영상별 모든 트랙 저장 |
| `select_subject.py` | 4.3 ① | 끊긴 트랙 합치기, 라벨을 붙일 피험자 1명 선택 |
| `features.py` | 4.3 ② | 정규화와 55차원 특징. 학습·실시간 공용 |
| `dataset.py` | 4.3 ③ | 프레임별 정답, 30프레임 창 |
| `models.py` | 4.3 ④ | TCNTE(메인), GRU, 1D-CNN |
| `augment.py` | 4.3 ⑤ | 좌우 반전, 좌표 흔들기, 속도 변화, 관절 지우기 |
| `train.py` | 4.3 ④ | 가중 Focal Loss 학습, 시드 3개 |
| `evaluate.py` | 6장 | 실시간 방식 프레임별 F1, 혼동행렬, 낮·밤 분리 |
| `infer.py` | 4.3 ⑧ | 구현팀용 `StreamClassifier` |
| `tests/` | 파일럿 | 가짜 관절 영상으로 각 단계 검증 |

## 실행

```bash
pip install -r requirements.txt
python -m unittest discover -s tests                       # 가짜 관절로 코드 검증

python download_data.py labels GMDCSA24 night              # 라벨, 영상 1.0 GiB, 밤 영상 목록
python extract_pose.py --dataset GMDCSA24                  # out/pose/GMDCSA24
python dataset.py --datasets GMDCSA24                      # out/data/GMDCSA24.npz
python train.py --datasets GMDCSA24 --model gru --out out/runs/e1_gru          # E1
python evaluate.py --run out/runs/e1_gru --datasets GMDCSA24 --split val

# E2: 데이터를 모두 받은 뒤
python download_data.py of-syn caucafall night
python extract_pose.py --dataset of-syn && python extract_pose.py --dataset caucafall
python dataset.py --datasets of-syn GMDCSA24 caucafall
for m in tcnte gru cnn; do python train.py --datasets of-syn GMDCSA24 caucafall --model $m --out out/runs/e2_$m; done
```

테스트 분할(`--split test`)은 모델과 설정을 다 정한 뒤 마지막에 한 번만 쓴다.

## 구현팀 연결

전달물은 `best.pt`, `features.py`, `models.py`, `infer.py`다.

```python
from infer import StreamClassifier
clf = StreamClassifier("out/runs/e2_tcnte/best.pt")
for frame in frames:                                   # 15fps로 공급
    for pid, kpts, box in pose_and_track(frame):       # kpts (17,3) = x, y, 신뢰도 / box = x1, y1, x2, y2 (픽셀)
        p_normal, p_falling, p_fallen = clf.step(pid, kpts, box)
    # 이번 프레임에 안 잡힌 사람은 clf.step(pid, None, None)
```

포즈 모델, 추적 설정, fps가 학습 때(YOLO26n-pose, `bytetrack.yaml`, 15fps)와 다르면 입력 분포가 달라지므로 다시 추출하고 학습해야 한다.

## 제안서에 없어 정한 것

- 영상 앞부분: 30프레임이 안 찬 창은 앞을 0으로 채우고 채운 프레임은 손실에서 뺀다.
- 사람을 못 찾은 프레임: 모든 관절이 빠진 것으로 처리한다 (좌표는 직전 값, 신뢰도 0). 한 번도 못 본 사람은 전부 0.
- 트랙 합치기 간격은 3초다 (제안서 1초). GMDCSA24에서 넘어지는 동안 1~2초 놓쳤다가 다시 잡히는 영상이 많았고, 검증 분할의 쓰러짐 구간 검출률이 60%에서 68%로 올랐다. ID가 바뀌는 순간의 5프레임 이내 겹침도 같은 사람으로 본다.
- 움직이는 사람이 있으면 제자리 검출(박스 중심 이동이 박스 긴 변의 15% 미만)은 피험자 후보에서 뺀다. GMDCSA24 피험자 2·3의 방에서 벽에 걸린 옷이 영상 내내 사람으로 잡혔다.
- 추적 설정은 제안서대로 `bytetrack.yaml` 기본값이다. 임계값을 낮춘 설정(conf 0.05, new_track_thresh 0.1)은 GMDCSA24에서 영상당 트랙이 1.8개에서 3.3개로 늘어 학습 영상 15개가 "주인공 불명"으로 빠졌고, 테스트 분할 쓰러짐 검출률도 82%에서 80%로 오르지 않았다.
- 낙상 구간이 없는 학습·검증 영상의 피험자: 테스트와 같이 가장 오래 잡힌 사람.
- 어떤 라벨 구간에도 속하지 않는 프레임은 정답을 모르므로 제외한다.
- 특징 범위: 각도는 90으로, 속도는 ±3에서 잘라 3으로, 박스 비율은 log2(w/h)를 ±2에서 잘라 2로 나눈다. 입력 표준화는 따로 하지 않는다.
- TCNTE에 위치 임베딩(30 x 32)을 넣었다. 파라미터는 20,891개다 (제안서 표기 약 2.2만).
- 조기 종료 patience 15. macro-F1은 정답에 나온 클래스만 평균한다.
- 관절 지우기 증강은 정규화된 좌표를 직전 값으로 고정한다 (실제 추출은 픽셀 좌표를 고정한 뒤 정규화하므로 근사).

## 지금까지의 결과 (GMDCSA24만, 검증 분할 47개 영상, 테스트 분할은 아직 쓰지 않음)

코드가 실제 데이터로 끝까지 도는지 확인한 E1이다. 학습 영상이 70개뿐이라 모델 비교 근거로 쓰기에는 작다.

| 모델 | 파라미터 | 검증 macro-F1 (시드 3개) |
|---|---|---|
| GRU | 170,499 | 0.784 ± 0.006 |
| TCNTE | 20,891 | 0.748 ± 0.013 |
| 1D-CNN | 58,947 | 0.748 ± 0.008 |

피험자 선택 결과: 학습 70/75, 검증 47/48, 테스트 35/37개 사용. 쓰러짐 구간에서 피험자가 잡힌 프레임은 학습 78%, 검증 68%, 테스트 82%.
