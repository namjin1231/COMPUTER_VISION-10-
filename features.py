import sys
from pathlib import Path
import numpy as np

FPS = 15
MIN_CONF = 0.3   # 이보다 낮은 관절은 결측
MAX_GAP = 2      # 2프레임까지는 보간, 3프레임 이상 끊기면 0

def fill_short_gaps(a, max_gap=MAX_GAP):
    """NaN이 연속 max_gap개 이하면 앞뒤 값으로 직선 보간. 나머지 NaN은 그대로 둔다."""
    a = a.astype(float).copy()
    bad = np.isnan(a)
    n, i = len(a), 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n:
            a[i:j] = np.interp(np.arange(i, j), [i - 1, j], [a[i - 1], a[j]])
        i = j
    return a

def compute_features(path):
    d = np.load(path)
    frames, kpts, boxes = d["frames"], d["kpts"], d["boxes"]
    F = int(frames.max()) + 1

    # 빈 프레임은 NaN으로 채운 전체 배열 만들기
    K = np.full((F, 17, 3), np.nan)
    K[frames] = kpts
    B = np.full((F, 4), np.nan)
    B[frames] = boxes

    # 신뢰도 낮은 관절은 결측 처리 후 짧은 구간만 보간
    low = ~(K[:, :, 2] >= MIN_CONF)
    K[low] = np.nan
    for j in range(17):
        for c in range(3):
            K[:, j, c] = fill_short_gaps(K[:, j, c])
    for c in range(4):
        B[:, c] = fill_short_gaps(B[:, c])

    w = B[:, 2] - B[:, 0]
    h = B[:, 3] - B[:, 1]
    cx = (B[:, 0] + B[:, 2]) / 2
    cy = (B[:, 1] + B[:, 3]) / 2
    ok = ~np.isnan(B).any(axis=1) & (h > 1)
    hs = np.where(ok, h, 1.0)
    miss = np.isnan(K[:, :, 0]) | np.isnan(K[:, :, 1])

    # 박스 중심을 빼고 박스 높이로 나눠서 정규화
    nx = (K[:, :, 0] - cx[:, None]) / hs[:, None]
    ny = (K[:, :, 1] - cy[:, None]) / hs[:, None]
    conf = np.nan_to_num(K[:, :, 2])
    joints = np.stack([nx, ny, conf], axis=2)
    joints[miss] = 0.0
    joints = np.nan_to_num(joints)

    # 몸통 각도: 어깨 중점 -> 엉덩이 중점. 서 있으면 0도, 누우면 약 90도
    s = (joints[:, 5, :2] + joints[:, 6, :2]) / 2
    hp = (joints[:, 11, :2] + joints[:, 12, :2]) / 2
    angle = np.degrees(np.arctan2(np.abs(hp[:, 0] - s[:, 0]), np.abs(hp[:, 1] - s[:, 1])))
    angle = np.where(~miss[:, [5, 6, 11, 12]].any(axis=1), angle, 0.0)

    # 박스 가로/세로 비율
    aspect = np.where(ok, w / hs, 0.0)

    # 세로 속도 (직전 프레임의 박스 높이 기준, 초당)
    def vel(a, valid):
        v = np.zeros(F)
        both = valid[1:] & valid[:-1]
        v[1:] = np.where(both, (a[1:] - a[:-1]) / hs[:-1] * FPS, 0.0)
        return np.nan_to_num(v)

    head_v = vel(K[:, 0, 1], ok & ~miss[:, 0])
    cy_v = vel(cy, ok)

    feats = np.zeros((F, 55))
    feats[:, :51] = joints.reshape(F, 51)
    feats[:, 51] = angle
    feats[:, 52] = head_v
    feats[:, 53] = aspect
    feats[:, 54] = cy_v
    feats[~ok] = 0.0          # 박스가 없는 프레임은 전부 0
    return feats

if __name__ == "__main__":
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    f = compute_features(src)
    np.save(out_dir / (src.stem + ".npy"), f.astype(np.float32))
    print(f"{src.stem}: 특징 {f.shape}")
    print("빈 프레임(전부 0):", np.where(~f.any(axis=1))[0].tolist())
    print("몸통 각도(10프레임마다):", [round(float(a)) for a in f[::10, 51]])
    print("박스 가로/세로 비율(10프레임마다):", [round(float(a), 2) for a in f[::10, 53]])