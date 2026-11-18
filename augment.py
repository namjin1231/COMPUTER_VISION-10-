"""⑤ 데이터 증강 (제안서 4.3 ⑤). 학습 때만 쓴다. 입력은 창 묶음 X (B, 30, 55).
  좌우 반전        x 부호를 뒤집고 좌우 관절 번호를 바꾼다. 각도·속도·박스 비율은 좌우와 무관
  좌표 흔들기      보이는 관절의 x, y에 작은 잡음
  속도 0.8~1.2배   dataset.gather가 창을 다시 뽑고, 여기서는 속도 특징에 같은 배율을 곱한다
  관절 1~4개 지우기  실제 추출과 같은 모양: 어느 시점부터 신뢰도 0, 좌표는 직전 값 유지
회전·크기 변환은 이미 계산된 몸통 각도와 값이 어긋나므로 쓰지 않는다.
"""
import numpy as np

from features import CY_V, HEAD_V

PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
SWAP = np.arange(17)
for _a, _b in PAIRS:
    SWAP[_a], SWAP[_b] = _b, _a
P = 0.5
SPEED = (0.8, 1.2)
JITTER = 0.01


def sample_speed(n, rng):
    """창마다 배속. 절반은 1(그대로)"""
    return np.where(rng.random(n) < P, rng.uniform(*SPEED, n), 1.0)


def scale_velocity(X, speed):
    X[:, :, [HEAD_V, CY_V]] = np.clip(X[:, :, [HEAD_V, CY_V]] * speed[:, None, None], -1, 1)
    return X


def flip(X):
    J = X[:, :, :51].reshape(len(X), X.shape[1], 17, 3)[:, :, SWAP].copy()
    J[..., 0] *= -1
    out = X.copy()
    out[:, :, :51] = J.reshape(len(X), X.shape[1], 51)
    return out


def jitter(X, rng, sigma=JITTER):
    J = X[:, :, :51].reshape(len(X), X.shape[1], 17, 3).copy()
    J[..., :2] += rng.normal(0, sigma, J[..., :2].shape) * (J[..., 2:3] > 0)
    out = X.copy()
    out[:, :, :51] = J.reshape(len(X), X.shape[1], 51)
    return out.astype(np.float32)


def drop_joints(x, rng, lo=1, hi=4):
    """창 하나 (T, 55). 관절 1~4개를 임의 시점부터 임의 길이만큼 지운다."""
    T = len(x)
    J = x[:, :51].reshape(T, 17, 3).copy()
    for j in rng.choice(17, size=int(rng.integers(lo, hi + 1)), replace=False):
        t0 = int(rng.integers(0, T))
        t1 = min(T, t0 + int(rng.integers(3, T + 1)))
        J[t0:t1, j, :2] = J[max(t0 - 1, 0), j, :2]          # 직전 값 유지
        J[t0:t1, j, 2] = 0.0
    out = x.copy()
    out[:, :51] = J.reshape(T, 51)
    return out


def augment_batch(X, speed, rng):
    """X는 이미 speed 배속으로 뽑힌 창. 나머지 증강을 각각 확률 0.5로 건다."""
    X = scale_velocity(X.copy(), speed)
    m = rng.random(len(X)) < P
    X[m] = flip(X[m])
    m = rng.random(len(X)) < P
    X[m] = jitter(X[m], rng)
    for i in np.where(rng.random(len(X)) < P)[0]:
        X[i] = drop_joints(X[i], rng)
    return X
