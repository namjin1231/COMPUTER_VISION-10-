"""피험자 1명의 관절 시퀀스 -> 프레임당 특징 55개 (제안서 4.3 ②)
학습(전체 영상 한 번에)과 실시간(프레임 하나씩)에서 같은 코드를 쓴다. 구현팀도 이 파일을 그대로 쓴다.

입력 (프레임 하나):
  kpts (17, 3)  관절 x, y(픽셀), 신뢰도. 사람을 못 찾은 프레임은 None
  box  (4,)     박스 x1, y1, x2, y2(픽셀). 사람을 못 찾은 프레임은 None

처리:
  1) 빠진 관절: 신뢰도 MIN_CONF(0.3) 미만이거나 사람을 못 찾은 프레임이면 좌표를 직전 프레임 값으로 채운다
     (실시간과 같은 조건: 미래 프레임을 쓰지 않는다). 신뢰도 채널은 원래 값 그대로 둔다(못 찾은 프레임은 0)
     박스도 같은 방식으로 직전 값 유지. 한 번도 본 적 없는 관절은 엉덩이 중심(정규화 후 0)에 둔다
  2) 정규화 (MediaPipe 방식): 엉덩이 중심을 빼고, max(몸통 길이 x 2.5, 중심에서 가장 먼 관절까지 거리)로 나눈다
     박스 높이로 나누면 누운 사람이 크게 부풀려지는 문제를 피한다

특징 55개:
  [0:51]  관절 17개 x (x, y, 신뢰도)   x, y는 정규화 좌표, y는 아래가 +
  51      몸통 각도 / 90                어깨 중점 -> 엉덩이 중점이 세로축과 이루는 각. 서 있음 0, 누움 1
  52      머리(코) 세로 속도             (몸 크기/초), ±VEL_CLIP에서 자르고 VEL_CLIP으로 나눔 -> ±1. 아래로 +
  53      박스 가로/세로 비율
  54      박스 중심 세로 속도            52와 같은 단위·스케일

사용:
  학습:   feats = compute_features(kpts_seq, box_seq)      # kpts_seq (F,17,3), 못 찾은 프레임은 NaN
  실시간: fs = FeatureStream();  f = fs.update(kpts, box)   # 매 프레임 (55,) 하나. 사람마다 FeatureStream 1개
  두 방식은 같은 값을 낸다 (compute_features가 FeatureStream을 그대로 돌린다)
"""
import numpy as np

FPS = 15
NFEAT = 55
MIN_CONF = 0.3        # 이보다 낮은 관절은 좌표를 믿지 않고 직전 값으로 채운다
TORSO_MULT = 2.5      # 정규화 크기 = max(몸통 길이 x 2.5, 가장 먼 관절까지 거리)
ANGLE_SCALE = 90.0    # 각도(도) / 90 -> 0 ~ 1
VEL_CLIP = 3.0        # 속도(몸 크기/초)를 ±3에서 자르고 3으로 나눔 -> ±1
L_SH, R_SH, L_HIP, R_HIP, NOSE = 5, 6, 11, 12, 0

class FeatureStream:
    """사람 1명분 상태(직전 관절·박스·크기)를 들고 프레임마다 특징 55개를 낸다"""
    def __init__(self, fps=FPS):
        self.fps = fps
        self.xy = None          # (17, 2) 직전 프레임까지 채운 픽셀 좌표 (본 적 없는 관절은 NaN)
        self.box = None         # (4,) 직전 박스
        self.prev = None        # 직전 프레임의 (코 y, 박스 중심 y, 정규화 크기)

    def update(self, kpts=None, box=None):
        found = kpts is not None and box is not None and np.all(np.isfinite(box))
        if found:
            kpts = np.asarray(kpts, dtype=np.float64)
            conf = np.nan_to_num(kpts[:, 2])
            good = (conf >= MIN_CONF) & np.isfinite(kpts[:, :2]).all(axis=1)
            xy = np.full((17, 2), np.nan) if self.xy is None else self.xy.copy()
            xy[good] = kpts[good, :2]                 # 믿을 만한 관절만 새 값, 나머지는 직전 값 유지
            self.xy, self.box = xy, np.asarray(box, dtype=np.float64)
        else:
            conf = np.zeros(17)                        # 못 찾은 프레임: 신뢰도 0, 좌표·박스는 직전 값
        if self.xy is None or self.box is None:        # 아직 한 번도 못 봤으면 전부 0
            self.prev = None
            return np.zeros(NFEAT, dtype=np.float32)

        xy = self.xy
        def mid(a, b):                                  # 한쪽만 보이면 그쪽 값, 둘 다 없으면 NaN
            p = xy[[a, b]]
            p = p[np.isfinite(p).all(axis=1)]
            return p.mean(axis=0) if len(p) else np.full(2, np.nan)
        hip, sh = mid(L_HIP, R_HIP), mid(L_SH, R_SH)
        x1, y1, x2, y2 = self.box
        bc = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if not np.all(np.isfinite(hip)):                # 엉덩이를 한 번도 못 봤으면 박스 중심을 기준으로
            hip = bc
        rel = xy - hip
        seen = np.isfinite(rel).all(axis=1)
        torso = np.linalg.norm(sh - hip) if np.all(np.isfinite(sh)) else 0.0
        far = np.linalg.norm(rel[seen], axis=1).max() if seen.any() else 0.0
        size = max(torso * TORSO_MULT, far, 1e-3)
        norm = np.where(seen[:, None], rel / size, 0.0)  # 본 적 없는 관절은 중심(0)

        # 몸통 각도: 어깨 중점 -> 엉덩이 중점이 세로축과 이루는 각 (0 ~ 90도)
        if np.all(np.isfinite(sh)) and torso > 0:
            d = hip - sh
            angle = np.degrees(np.arctan2(abs(d[0]), abs(d[1])))
        else:
            angle = 0.0

        w, h = x2 - x1, y2 - y1
        aspect = w / h if h > 1 else 0.0

        # 세로 속도: 직전 프레임의 몸 크기 기준, 초당. 직전 프레임이 없으면 0
        nose_y = xy[NOSE, 1]
        head_v = cy_v = 0.0
        if self.prev is not None:
            pn, pc, ps = self.prev
            if np.isfinite(nose_y) and np.isfinite(pn):
                head_v = (nose_y - pn) / ps * self.fps
            cy_v = (bc[1] - pc) / ps * self.fps
        self.prev = (nose_y, bc[1], size)

        f = np.empty(NFEAT)
        f[:51] = np.concatenate([norm, conf[:, None]], axis=1).reshape(-1)
        f[51] = angle / ANGLE_SCALE
        f[52] = np.clip(head_v, -VEL_CLIP, VEL_CLIP) / VEL_CLIP
        f[53] = aspect
        f[54] = np.clip(cy_v, -VEL_CLIP, VEL_CLIP) / VEL_CLIP
        return f.astype(np.float32)

def compute_features(kpts_seq, box_seq, fps=FPS):
    """(F,17,3), (F,4) -> (F,55). 사람을 못 찾은 프레임은 kpts·box를 NaN으로 채워서 넘긴다"""
    fs = FeatureStream(fps)
    out = np.zeros((len(kpts_seq), NFEAT), dtype=np.float32)
    for i, (k, b) in enumerate(zip(kpts_seq, box_seq)):
        found = np.all(np.isfinite(b)) and np.isfinite(k[:, 2]).any()
        out[i] = fs.update(k if found else None, b if found else None)
    return out
