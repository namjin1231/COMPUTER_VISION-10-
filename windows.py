"""관절 추출 결과 -> 학습용 창 (제안서 4.3 ③)

영상 1개마다:
  raw/{dataset}/{path}.npz  --피험자 1명 고르기-->  (F,17,3), (F,4)
  --features.py-->  (F,55)   --labels.py-->  (F,) 프레임별 정답
  --자르기-->  win/{dataset}/{path}.npz

창: 30프레임(2초), 5프레임 간격. 정답은 창마다 1개가 아니라 30프레임 각각 (제안서 '변경')
  - 마지막 창이 영상 끝에 안 맞으면 끝에 맞춘 창을 하나 더 만든다 (영상 끝 프레임도 학습에 들어가게)
  - 영상이 30프레임보다 짧으면 앞쪽을 0으로 채운다. 채운 프레임 정답은 제외(-1)
  - 30프레임 정답이 전부 제외(-1)인 창은 버린다
  - 제외(-1) 프레임은 학습 때 손실에서 뺀다 (train 쪽에서 처리)

저장 (영상 1개 = npz 1개):
  X (N,30,55) float32, y (N,30) int8, start (N,) 창 첫 프레임 번호, t0 (N,) 창 첫 프레임 시각(초)
  video (영상 키 "{dataset}/{path}"), split, daynight

사용: python windows.py <raw 폴더> <출력 폴더> [--only GMDCSA24]
  raw 폴더 아래 {dataset}/{path}.npz, split.json·daynight.json은 현재 폴더에서 읽는다
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from features import compute_features, NFEAT, FPS
from labels import load_segments, frame_labels, EXCLUDE, NCLS, CLASS_NAMES

WIN, STRIDE = 30, 5
CONFIG_OF = {"of-syn": "of-syn", "GMDCSA24": "gmdcsa24-cs", "caucafall": "caucafall-cs", "OOPS": "of-itw"}

def pick_subject(raw):
    """임시: 프레임마다 박스 신뢰도가 가장 높은 검출 1개. (F,17,3), (F,4), 못 찾은 프레임은 NaN
    TODO 제안서의 피험자 선택 규칙으로 바꾼다 (끊긴 트랙 1초 병합, 학습·검증은 낙상 구간 최다 트랙 + 1.5배 규칙,
    테스트는 최장 트랙, 잡힌 프레임 50% 미만 제외). 추출 형식이 정해지면 작업"""
    F = len(raw["t"])
    K, B, best = np.full((F, 17, 3), np.nan), np.full((F, 4), np.nan), np.zeros(F)
    for f, k, b, c in zip(raw["det_f"], raw["det_kpts"], raw["det_box"], raw["det_conf"]):
        if c > best[f]:
            best[f], K[f], B[f] = c, k, b
    return K, B

def make_windows(feat, y):
    """(F,55), (F,) -> X (N,30,55), y (N,30), start (N,)"""
    F = len(feat)
    if F < WIN:                                     # 짧은 영상: 앞쪽 0 채움, 정답은 제외
        pad = WIN - F
        feat = np.concatenate([np.zeros((pad, feat.shape[1]), feat.dtype), feat])
        y = np.concatenate([np.full(pad, EXCLUDE, y.dtype), y])
        starts, offset = [0], -pad
    else:
        starts = list(range(0, F - WIN + 1, STRIDE))
        if starts[-1] != F - WIN:
            starts.append(F - WIN)                  # 끝에 맞춘 창 하나 더
        offset = 0
    X, Y, S = [], [], []
    for s in starts:
        yy = y[s:s + WIN]
        if (yy == EXCLUDE).all():
            continue
        X.append(feat[s:s + WIN]); Y.append(yy); S.append(s + offset)
    if not X:
        return np.zeros((0, WIN, NFEAT), np.float32), np.zeros((0, WIN), np.int8), np.zeros(0, int)
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.int8), np.array(S)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*", help="이 데이터셋만 (예: GMDCSA24)")
    args = ap.parse_args()
    raw_root, out_root = Path(args.raw), Path(args.out)
    split = {k: sp for sp, ks in json.load(open("split.json")).items() for k in ks}
    daynight = json.load(open("daynight.json"))

    seg_cache, stats = {}, collections.defaultdict(lambda: np.zeros(NCLS + 1, int))
    nwin, skipped = collections.Counter(), []
    for f in sorted(raw_root.rglob("*.npz")):
        key = f.relative_to(raw_root).with_suffix("").as_posix()      # "GMDCSA24/Subject_1/Fall/01"
        ds, path = key.split("/", 1)
        if args.only and ds not in args.only:
            continue
        if key not in split:
            skipped.append((key, "split.json에 없음")); continue
        if ds not in seg_cache:
            seg_cache[ds] = load_segments(CONFIG_OF[ds])[0]
        raw = np.load(f)
        K, B = pick_subject(raw)
        feat = compute_features(K, B)
        y = frame_labels(seg_cache[ds][path], len(feat))
        X, Y, S = make_windows(feat, y)
        dst = (out_root / key).with_suffix(".npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, X=X, y=Y, start=S, t0=S / FPS, video=key, split=split[key], daynight=daynight[key])
        sp = split[key]
        nwin[sp] += len(Y)
        stats[sp] += np.bincount(Y.reshape(-1) + 1, minlength=NCLS + 1)   # [제외, 0, 1, 2]
    for sp in ("train", "validation", "test"):
        if nwin[sp]:
            c = stats[sp]
            print(f"{sp}: 창 {nwin[sp]}개, 프레임 정답 " +
                  ", ".join(f"{n} {v}" for n, v in zip(CLASS_NAMES, c[1:])) + f", 제외 {c[0]}")
    for k, why in skipped:
        print("건너뜀:", k, why)
