"""학습용 창 불러오기 (모델 담당용). windows.py가 만든 win/{dataset}/{path}.npz를 분할별로 모은다.

  from win_data import load_windows
  d = load_windows("out/win_v", "train")                 # 모든 데이터셋
  d = load_windows("out/win_v", "test", datasets=["GMDCSA24"], daynight="night")
  d["X"]  (N, 30, 55) float32   특징 (features.py)
  d["y"]  (N, 30)     int64     프레임별 정답 0 정상 / 1 넘어지는 중 / 2 쓰러짐 / -1 제외
  d["video"], d["start"], d["t0"], d["daynight"], d["dataset"]  (N,)  창마다 메타 정보

제외(-1) 프레임은 손실과 점수에서 뺀다:
  mask = y >= 0
  loss = (frame_loss * mask).sum() / mask.sum()         # frame_loss: (B, 30)
"""
from pathlib import Path
import numpy as np

def load_windows(win_root, split, datasets=None, daynight=None):
    Xs, ys, meta = [], [], {k: [] for k in ("video", "start", "t0", "daynight", "dataset")}
    for f in sorted(Path(win_root).rglob("*.npz")):
        d = np.load(f)
        if str(d["split"]) != split:
            continue
        video = str(d["video"]); ds = video.split("/", 1)[0]
        if datasets and ds not in datasets:
            continue
        if daynight and str(d["daynight"]) != daynight:
            continue
        n = len(d["y"])
        if n == 0:
            continue
        Xs.append(d["X"]); ys.append(d["y"])
        meta["video"] += [video] * n
        meta["start"].append(d["start"]); meta["t0"].append(d["t0"])
        meta["daynight"] += [str(d["daynight"])] * n
        meta["dataset"] += [ds] * n
    if not Xs:
        raise ValueError(f"'{split}' 창이 없음: {win_root} (datasets={datasets}, daynight={daynight})")
    return {
        "X": np.concatenate(Xs).astype(np.float32),
        "y": np.concatenate(ys).astype(np.int64),
        "video": np.array(meta["video"]),
        "start": np.concatenate(meta["start"]),
        "t0": np.concatenate(meta["t0"]),
        "daynight": np.array(meta["daynight"]),
        "dataset": np.array(meta["dataset"]),
    }

if __name__ == "__main__":
    import sys
    from labels import CLASS_NAMES, NCLS
    root = sys.argv[1] if len(sys.argv) > 1 else "out/win_v"
    for sp in ("train", "validation", "test"):
        try:
            d = load_windows(root, sp)
        except ValueError as e:
            print(e); continue
        y = d["y"]
        c = np.bincount(y[y >= 0], minlength=NCLS)
        print(f"{sp}: 창 {len(y)}개, 영상 {len(set(d['video']))}개, 프레임 정답 "
              + ", ".join(f"{n} {v}" for n, v in zip(CLASS_NAMES, c)) + f", 제외 {(y < 0).sum()}")
