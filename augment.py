"""창 하나(30, 55)에 거는 증강. 학습 때만 쓴다.
특징 배치: [0:51] = 관절 17개 x (x, y, 신뢰도), 51 몸통 각도/90, 52 머리 세로 속도(±1), 53 박스 가로/세로 비율, 54 박스 중심 세로 속도(±1)
회전·크기 변환은 몸통 각도와 값이 어긋나서 쓰지 않는다.
"""
import numpy as np

# COCO 17 관절의 좌우 짝: 눈, 귀, 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
SWAP = list(range(17))
for a, b in PAIRS:
    SWAP[a], SWAP[b] = b, a

def flip(w):
    j = w[:, :51].reshape(-1, 17, 3)[:, SWAP].copy()
    j[:, :, 0] *= -1                       # 박스 중심 기준 좌표라 x 부호만 뒤집으면 된다
    out = w.copy()
    out[:, :51] = j.reshape(-1, 51)        # 각도(절댓값), 속도, 비율은 좌우와 무관
    return out

def jitter(w, rng, sigma=0.01):
    j = w[:, :51].reshape(-1, 17, 3).copy()
    seen = j[:, :, 2] > 0                  # 못 찾은 프레임의 관절(신뢰도 0, 직전 좌표)은 건드리지 않는다
    j[:, :, :2] += rng.normal(0, sigma, j[:, :, :2].shape) * seen[:, :, None]
    out = w.copy()
    out[:, :51] = j.reshape(-1, 51)
    return out

def speed(w, rng, lo=0.8, hi=1.2):
    f = rng.uniform(lo, hi)
    n = len(w)
    t = np.clip(np.arange(n) * f, 0, n - 1)   # f>1이면 빨라지고, 끝을 넘으면 마지막 프레임 유지
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    a = (t - i0)[:, None]
    out = w[i0] * (1 - a) + w[i1] * a
    out[:, [52, 54]] = np.clip(out[:, [52, 54]] * f, -1, 1)   # 속도 특징은 배속만큼 같이 바뀐다 (features.py와 같이 ±1)
    return out

def drop_joints(w, rng, lo=1, hi=4):
    """features.py의 빠진 관절과 같은 모양: 신뢰도 0, 좌표는 직전 값 유지.
    창 전체에서 지우므로 창 첫 프레임 좌표로 고정한다 (그 전부터 못 본 것처럼)"""
    k = int(rng.integers(lo, hi + 1))
    js = rng.choice(17, size=k, replace=False)
    j = w[:, :51].reshape(-1, 17, 3).copy()
    j[:, js, :2] = j[0, js, :2]
    j[:, js, 2] = 0.0
    out = w.copy()
    out[:, :51] = j.reshape(-1, 51)
    return out

def augment(w, rng, p=0.5):
    for fn in (flip, lambda a: jitter(a, rng), lambda a: speed(a, rng), lambda a: drop_joints(a, rng)):
        if rng.random() < p:
            w = fn(w)
    return w
