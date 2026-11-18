"""⑧ 실시간 방식 예측 (제안서 4.3 ⑧). 매 프레임, 그 프레임까지의 최근 30프레임만으로 판단한다.

구현팀 연결 (best.pt, features.py, models.py, infer.py만 있으면 된다)

    from infer import StreamClassifier
    clf = StreamClassifier("best.pt")
    for frame in frames:                                  # 15fps로 공급
        for pid, kpts, box in pose_and_track(frame):      # kpts (17,3) = x, y, 신뢰도 / box = x1, y1, x2, y2 (픽셀)
            p = clf.step(pid, kpts, box)                  # [P(정상), P(넘어지는 중), P(쓰러짐)]
        # 이번 프레임에 안 잡힌 사람은 clf.step(pid, None, None)을 불러 시간을 맞춘다

포즈 모델·추적 설정·fps가 학습 때(YOLO26n-pose, bytetrack.yaml, 15fps)와 다르면 입력 분포가 달라진다.
"""
from collections import deque

import numpy as np
import torch

from features import N_FEAT, FeatureState
from models import WIN, load_ckpt


@torch.no_grad()
def _last_probs(model, X, batch=4096):
    """X (N, 30, 55) -> 각 창의 마지막 프레임 확률 (N, 3)"""
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(X[i:i + batch]))[:, -1], -1).numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def predict_stream(model, feat):
    """영상 하나의 특징 (T, 55) -> 프레임별 확률 (T, 3). 프레임 t는 t-29..t만 본다 (앞이 모자라면 0)."""
    pad = np.concatenate([np.zeros((WIN - 1, feat.shape[1]), np.float32), feat.astype(np.float32)])
    X = np.lib.stride_tricks.sliding_window_view(pad, WIN, axis=0).transpose(0, 2, 1)
    return _last_probs(model, np.ascontiguousarray(X))


def predict_corpus(model, corpus, chunk=8192):
    """dataset.Corpus 전체를 실시간 방식으로 예측 -> (전체 프레임 수, 3). corpus.y와 같은 순서"""
    from dataset import gather
    n = len(corpus.y)
    vid = np.repeat(np.arange(len(corpus)), np.diff(corpus.start))
    end = np.arange(n) - corpus.start[vid]
    out = np.zeros((n, 3), np.float32)
    for i in range(0, n, chunk):
        X, _ = gather(corpus, vid[i:i + chunk], end[i:i + chunk])
        out[i:i + chunk] = _last_probs(model, X)
    return out


class StreamClassifier:
    """사람(pid)별로 특징 상태와 30프레임 버퍼를 들고, 프레임이 들어올 때마다 확률 3개를 낸다."""

    def __init__(self, ckpt):
        self.model, self.info = load_ckpt(ckpt)
        self.people = {}

    def step(self, pid, kpts, box):
        if pid not in self.people:
            self.people[pid] = (FeatureState(), deque(maxlen=WIN))
        state, buf = self.people[pid]
        buf.append(state.step(kpts, box))
        X = np.zeros((1, WIN, N_FEAT), np.float32)
        X[0, WIN - len(buf):] = np.stack(buf)
        return _last_probs(self.model, X)[0]

    def drop(self, pid):
        """화면에서 사라진 사람의 상태를 지운다"""
        self.people.pop(pid, None)
