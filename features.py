"""② 정규화와 특징 계산 (제안서 4.3 ②). 학습과 실시간이 이 파일 하나를 같이 쓴다.

프레임당 55차원
  [0:51]  관절 17개 x (x, y, 신뢰도). 엉덩이 중심을 빼고 max(몸통 길이 x 2.5, 중심에서 가장 먼 관절 거리)로 나눔
  [51]    몸통 각도 / 90        서 있으면 0, 누우면 1
  [52]    머리 세로 속도        (크기 단위/초)를 ±3에서 자르고 3으로 나눔. 아래로 움직이면 +
  [53]    박스 가로/세로 비율   log2(w/h)를 ±2에서 자르고 2로 나눔. 정사각형이면 0, 누우면 +
  [54]    박스 중심 세로 속도   [52]와 같은 단위

모든 계산은 과거 프레임만 쓴다. FeatureState.step()을 한 프레임씩 부르는 것이 유일한 계산 경로이고,
compute_features()는 그것을 영상 전체에 반복할 뿐이라 학습용 특징과 실시간 특징이 항상 같다.
"""
import numpy as np

FPS = 15
N_FEAT = 55
MIN_CONF = 0.3       # 이보다 낮은 관절은 빠진 것으로 보고 직전 값을 유지 (신뢰도는 0)
TORSO_MULT = 2.5
VEL_CLIP = 3.0
ASPECT_CLIP = 2.0
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
HEAD = [0, 1, 2, 3, 4]
ANGLE, HEAD_V, ASPECT, CY_V = 51, 52, 53, 54


def _mid(xy, seen, a, b):
    """두 관절의 중점. 하나만 본 적 있으면 그 관절, 둘 다 없으면 None"""
    if seen[a] and seen[b]:
        return (xy[a] + xy[b]) / 2
    if seen[a]:
        return xy[a]
    if seen[b]:
        return xy[b]
    return None


class FeatureState:
    """한 사람의 특징을 프레임 순서대로 계산한다. 사람(추적 ID)마다 하나씩 만든다."""

    def __init__(self, fps=FPS):
        self.fps = fps
        self.xy = np.zeros((17, 2))            # 관절별 마지막으로 믿을 만했던 좌표 (픽셀)
        self.seen = np.zeros(17, bool)         # 한 번이라도 잡힌 관절
        self.box = None                        # 마지막 박스
        self.prev = None                       # 직전 프레임의 (머리 y, 박스 중심 y)

    def step(self, kpts, box):
        """kpts (17, 3) = x, y, 신뢰도 / box (4,) = x1, y1, x2, y2. 사람을 못 찾은 프레임은 둘 다 None.
        못 찾은 프레임은 모든 관절이 빠진 것과 같이 처리한다 (좌표 유지, 신뢰도 0)."""
        out = np.zeros(N_FEAT, np.float32)
        conf = np.zeros(17)
        if kpts is not None:
            kpts = np.asarray(kpts, float)
            good = np.isfinite(kpts).all(1) & (kpts[:, 2] >= MIN_CONF)
            self.xy[good] = kpts[good, :2]
            self.seen |= good
            conf[good] = kpts[good, 2]
            self.box = np.asarray(box, float)
        if self.box is None or not self.seen.any():
            self.prev = None
            return out                         # 아직 한 번도 못 본 사람: 전부 0

        xy, seen, b = self.xy, self.seen, self.box
        w, h = max(b[2] - b[0], 1e-6), max(b[3] - b[1], 1e-6)
        hip = _mid(xy, seen, L_HIP, R_HIP)
        sh = _mid(xy, seen, L_SH, R_SH)
        center = hip if hip is not None else np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        torso, angle = 0.0, 0.0
        if hip is not None and sh is not None:
            d = hip - sh
            torso = float(np.hypot(d[0], d[1]))
            if torso > 1e-6:
                angle = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1])))) / 90.0
        far = float(np.linalg.norm(xy[seen] - center, axis=1).max())
        scale = max(torso * TORSO_MULT, far)
        if scale < 1e-6:
            scale = max(w, h) / 2

        n = (xy - center) / scale
        n[~seen] = 0.0
        out[:51] = np.concatenate([n, conf[:, None]], 1).reshape(51)

        head_seen = [j for j in HEAD if seen[j]]
        head_y = float(xy[head_seen, 1].mean()) if head_seen else None
        cy = float((b[1] + b[3]) / 2)
        head_v = cy_v = 0.0
        if self.prev is not None:
            if head_y is not None and self.prev[0] is not None:
                head_v = (head_y - self.prev[0]) / scale * self.fps
            cy_v = (cy - self.prev[1]) / scale * self.fps
        self.prev = (head_y, cy)

        out[ANGLE] = angle
        out[HEAD_V] = np.clip(head_v, -VEL_CLIP, VEL_CLIP) / VEL_CLIP
        out[ASPECT] = np.clip(np.log2(w / h), -ASPECT_CLIP, ASPECT_CLIP) / ASPECT_CLIP
        out[CY_V] = np.clip(cy_v, -VEL_CLIP, VEL_CLIP) / VEL_CLIP
        return out


def compute_features(kpts, box, valid, fps=FPS):
    """영상 전체: kpts (T, 17, 3), box (T, 4), valid (T,) 사람이 잡힌 프레임 -> (T, 55)"""
    st = FeatureState(fps)
    out = np.zeros((len(valid), N_FEAT), np.float32)
    for t in range(len(valid)):
        out[t] = st.step(kpts[t], box[t]) if valid[t] else st.step(None, None)
    return out
