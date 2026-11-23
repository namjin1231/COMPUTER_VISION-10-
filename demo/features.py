"""features.py: 관절 (T,17,2),(T,17),(T,4) -> 특징 (T,55). 학습과 실시간 시스템에서 똑같이 쓴다. 미래 프레임을 쓰지 않는다."""
import numpy as np

CONF_T = 0.3
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _ffill(kpts, conf):
    k = kpts.astype(np.float32).copy()
    c = conf.astype(np.float32).copy()
    last = np.zeros((17, 2), np.float32)
    for t in range(len(k)):
        ok = c[t] >= CONF_T
        last[ok] = k[t, ok]
        k[t, ~ok] = last[~ok]          # 직전 값 (한 번도 안 잡혔으면 0)
    return k, c


def norm_center_scale(k, box, seen):
    """한 프레임의 중심 (2,)과 크기 (스칼라). seen: 지금까지 한 번이라도 잡힌 관절"""
    bw, bh = box[2] - box[0], box[3] - box[1]
    if seen[5] and seen[6] and seen[11] and seen[12]:
        hip = (k[11] + k[12]) / 2
        sh = (k[5] + k[6]) / 2
        torso = np.linalg.norm(sh - hip)
        maxd = np.linalg.norm(k[seen] - hip, axis=1).max()
        return hip, max(torso * 2.5, maxd, 1e-6)
    ctr = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
    return ctr, max(bw, bh, 1e-6)


def compute_features(kpts, conf, box):
    """(T,17,2) 픽셀, (T,17), (T,4) xyxy -> (T,55) float32. 미래 프레임을 쓰지 않는다."""
    T = len(kpts)
    k, c = _ffill(kpts, conf)
    out = np.zeros((T, 55), np.float32)
    seen = np.zeros(17, bool)
    prev_nose = prev_bc = None
    for t in range(T):
        seen |= c[t] >= CONF_T
        b = box[t]
        if not np.any(b):
            # 사람이 안 잡힌 프레임(박스 0): 특징 전부 0으로 두고, 다음 프레임 속도도 0이 되게 직전 값을 지운다.
            # (예전에는 크기가 1e-6이 되어 값이 수억까지 커졌음. 이 프레임들의 정답은 원래 -1)
            prev_nose = prev_bc = None
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        ctr, s = norm_center_scale(k[t], b, seen)
        xy = (k[t] - ctr) / s
        xy[~seen] = 0
        out[t, :51] = np.concatenate([xy, c[t][:, None]], 1).reshape(-1)
        if seen[5] and seen[6] and seen[11] and seen[12]:
            v = (k[t, 5] + k[t, 6]) / 2 - (k[t, 11] + k[t, 12]) / 2       # 엉덩이 -> 어깨
            ang = np.degrees(np.arctan2(abs(v[0]), -v[1] + 1e-9))          # 0 = 똑바로 섬 (영상 y는 아래가 +)
            out[t, 51] = np.clip(ang, 0, 180) / 90
        nose = k[t, 0]
        bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        if prev_nose is not None:
            out[t, 52] = (nose[1] - prev_nose[1]) / s
            out[t, 54] = (bc[1] - prev_bc[1]) / s
        out[t, 53] = bw / max(bh, 1e-6)
        prev_nose, prev_bc = nose, bc
    return out


def flip_features(X):
    """(...,55) 좌우 반전"""
    X = X.copy()
    J = X[..., :51].reshape(*X.shape[:-1], 17, 3)
    J[..., 0] *= -1
    for a, b in FLIP_PAIRS:
        J[..., [a, b], :] = J[..., [b, a], :]
    X[..., :51] = J.reshape(*X.shape[:-1], 51)
    return X
