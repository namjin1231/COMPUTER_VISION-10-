import sys
from pathlib import Path
import numpy as np

WIN, STRIDE, TAIL = 30, 5, 10
LABEL_MAP = {-1: 0, 0: 1, 1: 2}   # 정상, 넘어지는 중, 쓰러짐

def load_labels(csv_path, seq):
    """CSV에서 seq(예: 'fall-01')의 {원본 프레임 번호: 라벨} 딕셔너리"""
    lab = {}
    for line in open(csv_path):
        p = line.strip().split(",")
        if p[0] == seq:
            lab[int(p[1])] = int(p[2])
    return lab

def frame_labels(lab, n):
    """15fps 특징 i번째는 원본 프레임 2i+1. 라벨이 없는 프레임은 가장 가까운 번호의 라벨을 쓴다."""
    keys = np.array(sorted(lab))
    out = []
    for i in range(n):
        k = 2 * i + 1
        near = int(keys[np.abs(keys - k).argmin()])
        out.append(LABEL_MAP[lab[near]])
    return np.array(out)

def make_windows(feat, y):
    X, Y, S = [], [], []
    for s in range(0, len(feat) - WIN + 1, STRIDE):
        tail = y[s + WIN - TAIL : s + WIN]
        Y.append(np.bincount(tail, minlength=3).argmax())
        X.append(feat[s : s + WIN])
        S.append(s)
    if not X:
        return np.zeros((0, WIN, feat.shape[1])), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.stack(X), np.array(Y), np.array(S)

if __name__ == "__main__":
    feat_path, csv_path, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = feat_path.stem.split("-cam0")[0]          # 'fall-01-cam0-rgb' -> 'fall-01'
    feat = np.load(feat_path)
    y = frame_labels(load_labels(csv_path, seq), len(feat))
    X, Y, S = make_windows(feat, y)
    np.savez(out_dir / f"{seq}.npz", X=X.astype(np.float32), y=Y,
             start=S, vid=np.array([seq] * len(Y)))
    print(f"{seq}: X {X.shape}, y {Y.shape}")
    print("클래스별 개수 [정상, 넘어지는 중, 쓰러짐]:", np.bincount(Y, minlength=3).tolist())
    print("창 라벨 순서:", Y.tolist())