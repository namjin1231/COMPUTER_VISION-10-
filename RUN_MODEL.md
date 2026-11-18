# 학습된 분류 모델 실행하기

영상이나 웹캠에 학습된 분류 모델(TCNTE)을 돌려, 사람마다 **정상 / 넘어지는 중 / 쓰러짐** 판정을 화면에 그려 본다.
학습 방법과 성능은 [MODEL_TRAINING.md](MODEL_TRAINING.md)를 참고한다.

## 준비

```bash
pip install -r requirements.txt
```

다음 두 파일이 있어야 한다.
- **`out/runs/e2_tcnte/best.pt`:** 학습된 모델이다. GitHub에는 올라가지 않으니 학습팀에게 따로 받아 이 위치에 둔다.
- **`yolo26n-pose.pt`:** 관절 추출 모델이다. 없으면 처음 실행할 때 자동으로 받는다.

## 실행

**영상 파일**
```bash
python demo.py 영상.mp4 --device mps
```
결과 영상은 `out/demo/영상.mp4`에 저장된다. 맥은 `open out/demo/영상.mp4`로 열면 된다.

**처리하면서 창으로 보기** (q로 종료)
```bash
python demo.py 영상.mp4 --show --device mps
```

**웹캠** (q로 종료)
```bash
python demo.py 0 --show --device mps
```
맥에서는 터미널 앱에 카메라 권한이 있어야 한다. 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 켠다.

**다른 모델로 비교**
```bash
python demo.py 영상.mp4 --ckpt out/runs/e2_gru/best.pt --device mps
```

`--device`는 맥 GPU면 `mps`, NVIDIA GPU면 `cuda:0`, 둘 다 없으면 `cpu`로 둔다.

## 결과 보는 법

| 표시 | 뜻 |
|---|---|
| 초록 박스·관절 | Normal (정상) |
| 주황 박스·관절 | Falling (넘어지는 중) |
| 빨강 박스·관절 | Fallen (쓰러짐) |
| `#2 Fallen 0.55` | 추적 번호 2번 사람, 판정, 그 확률 |
| 박스 오른쪽 막대 3개 | 정상, 넘어지는 중, 쓰러짐 확률 |

실행이 끝나면 터미널에 사람별 판정 구간이 출력된다.

```
  #2: Falling 1.9~2.7s, Fallen 3.5~6.7s
```

## 참고

- **분류 모델 출력만 보여 준다.** 경보 규칙(쓰러짐이 N초 이어지면 경보 등)은 없어서, 프레임마다 판정이 잠깐씩 흔들릴 수 있다.
- **고정된 카메라가 가장 잘 맞는다.** 학습 데이터처럼 고정된 카메라로 사람 전신이 보일 때 가장 잘 맞는다. 휴대폰 촬영이나 카메라가 움직이는 영상에서는 정확도가 크게 떨어진다.
- **웹캠은 처리 속도를 확인한다.** 처리가 초당 15프레임보다 느리면 실행 끝에 경고가 나온다. 이때는 동작이 실제보다 빠르게 보여 판정이 틀어질 수 있으니 `--show`를 빼고 실행해 본다.
- **예제 영상:** 데이터를 받아 두었다면 아래 영상으로 바로 시험할 수 있다.
  ```bash
  python demo.py data/omnifall/GMDCSA24/video/Subject_4/Fall/03.mp4 --device mps   # 낙상
  python demo.py data/omnifall/GMDCSA24/video/Subject_4/ADL/01.mp4 --device mps    # 일상 동작
  ```
