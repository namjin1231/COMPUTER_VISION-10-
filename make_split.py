"""영상(시퀀스) 단위로 학습/검증/테스트를 나눠 split.json으로 고정한다.
같은 영상의 창이 두 쪽에 섞이지 않게 하려고 창이 아니라 영상 이름으로 나눈다.
낙상(fall)·일상(adl)을 따로 70/15/15로 나눠서 각 분할에 둘 다 들어가게 한다.

사용: python make_split.py [창 폴더=out/win] [출력=split.json]
"""
import json
import sys
from pathlib import Path
import numpy as np
from dataset import kind_of

SEED = 0
RATIO = (0.70, 0.15, 0.15)

def split_names(names, rng):
    names = list(names)
    rng.shuffle(names)
    n = len(names)
    n_tr = round(n * RATIO[0])
    n_va = round(n * RATIO[1])
    return names[:n_tr], names[n_tr:n_tr + n_va], names[n_tr + n_va:]

if __name__ == "__main__":
    win_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/win")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "split.json")
    seqs = sorted(p.stem for p in win_dir.glob("*.npz"))
    rng = np.random.default_rng(SEED)
    split = {"train": [], "val": [], "test": []}
    for kind in ("fall", "adl"):
        tr, va, te = split_names([s for s in seqs if kind_of(s) == kind], rng)
        split["train"] += tr; split["val"] += va; split["test"] += te
    for k in split:
        split[k].sort()
    out.write_text(json.dumps(split, indent=1, ensure_ascii=False))
    print({k: len(v) for k, v in split.items()}, "->", out)
