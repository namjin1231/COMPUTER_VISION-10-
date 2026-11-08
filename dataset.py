import sys
from pathlib import Path
import numpy as np

WIN, STRIDE, TAIL = 30, 5, 10
NCLS = 3   # 0 정상, 1 넘어지는 중, 2 쓰러짐
CLASS_NAMES = ["정상", "넘어지는 중", "쓰러짐"]
EXCLUDE = -1   # 학습·평가에서 빼는 프레임
# UR Fall CSV 라벨: -1 서 있음, 0 전환 구간, 1 누움
# 낙상 영상의 누움 = 쓰러짐. 일상 영상의 누움은 누운 모습만으로는 쓰러짐과 구별이 안 돼서 제외한다.
# ('일부러 누움' 클래스는 뺐음. 일상 영상의 전환 구간은 정상)
LABEL_MAPS = {
    "fall": {-1: 0, 0: 1, 1: 2},
    "adl":  {-1: 0, 0: 0, 1: EXCLUDE},
}

def kind_of(seq):
    """'fall-01' -> 'fall', 'adl-05' -> 'adl'"""
    return "fall" if seq.startswith("fall") else "adl"

def load_labels(csv_path, seq):
    """CSV에서 seq(예: 'fall-01')의 {원본 프레임 번호: 라벨} 딕셔너리"""
    lab = {}
    for line in open(csv_path):
        p = line.strip().split(",")
        if p[0] == seq:
            lab[int(p[1])] = int(p[2])
    return lab

def frame_labels(lab, n, kind):
    """15fps 특징 i번째는 원본 프레임 2i+1. 라벨이 없는 프레임은 가장 가까운 번호의 라벨을 쓴다."""
    keys = np.array(sorted(lab))
    m = LABEL_MAPS[kind]
    out = []
    for i in range(n):
        k = 2 * i + 1
        near = int(keys[np.abs(keys - k).argmin()])
        out.append(m[lab[near]])
    return np.array(out)

def make_windows(feat, y):
    """30프레임 창을 5프레임 간격으로 자른다.
    영상 앞쪽은 0으로 채워서, 창 끝이 실제 프레임 STRIDE-1(=4)번부터 오게 한다.
    (실시간 버퍼가 아직 덜 찬 초기 상태와 같은 모양. 0 = 사람을 못 찾은 프레임과 같은 표현)
    창 끝 위치는 4, 9, 14, ... 로 기존(0 채우기 전) 창들과 같은 격자를 쓴다.
    S(start)는 창 첫 프레임 번호라 앞쪽이 채워진 창은 음수가 된다.
    라벨은 창 끝 TAIL(10)프레임 중 실제 프레임들의 다수결. 다수결 1위가 EXCLUDE면 그 창은 만들지 않는다."""
    n, d = feat.shape
    pad = np.concatenate([np.zeros((WIN - 1, d), feat.dtype), feat])   # pad[i + WIN - 1] = feat[i]
    X, Y, S = [], [], []
    for e in range(STRIDE - 1, n, STRIDE):                  # e = 창의 마지막 실제 프레임
        tail = y[max(0, e - TAIL + 1) : e + 1]
        lab = np.bincount(tail + 1, minlength=NCLS + 1).argmax() - 1   # +1: EXCLUDE(-1)도 셀 수 있게
        if lab == EXCLUDE:
            continue
        Y.append(lab)
        X.append(pad[e : e + WIN])                           # 실제 프레임 e-WIN+1 ~ e
        S.append(e - WIN + 1)
    if not X:
        return np.zeros((0, WIN, d)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.stack(X), np.array(Y), np.array(S)

if __name__ == "__main__":
    feat_path, csv_path, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = feat_path.stem.split("-cam0")[0]          # 'fall-01-cam0-rgb' -> 'fall-01'
    feat = np.load(feat_path)
    y = frame_labels(load_labels(csv_path, seq), len(feat), kind_of(seq))
    X, Y, S = make_windows(feat, y)
    np.savez(out_dir / f"{seq}.npz", X=X.astype(np.float32), y=Y,
             start=S, vid=np.array([seq] * len(Y)))
    print(f"{seq}: X {X.shape}, y {Y.shape}")
    print(f"클래스별 개수 {CLASS_NAMES}:", np.bincount(Y, minlength=NCLS).tolist())
    print("창 라벨 순서:", Y.tolist())