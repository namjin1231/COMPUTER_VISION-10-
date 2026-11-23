"""5단계: 특징 계산 -> 영상별 시퀀스 저장 (학습할 때 여기서 2초 조각을 잘라 쓴다).

사용법: python 05_build_sequences.py
결과: seq/train.pkl, seq/val.pkl, seq/test.pkl, seq/itw.pkl (OOPS), seq/meta.json, logs/seq_stats.csv
"""
import json, pickle
import numpy as np
import pandas as pd
from common import POSE_SEL, SEQ, LOGS, W, STAGED, append_changes
from features import compute_features

STEP = 5


def main():
    import argparse
    from common import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--itw", action="store_true",
                    help="E6: pose_sel_itw/ 의 OOPS train->train, val->val 을 더해 seq_itw/ 에 따로 저장 (OOPS test는 그대로 itw)")
    ap.add_argument("--extra-sel", default=None,
                    help="E7: 이 pose_sel 폴더의 데이터를 더함. train->train(반복), val->val_extra, test->test_extra (OmniFall val·test는 그대로)")
    ap.add_argument("--out", default=None, help="시퀀스 출력 폴더 (예: seq_e7). --extra-sel과 같이 씀")
    ap.add_argument("--extra-share", type=float, default=0.25, help="--extra-sel 학습 조각 비중 목표 (반복 횟수를 정함)")
    ap.add_argument("--frac-repeat", action="store_true",
                    help="2차(_r2): AI Hub 반복 횟수를 소수(0.01 단위)로 정해 비중을 목표에 맞춤 (06이 소수 부분만큼 고정된 일부 조각을 씀). 기본 꺼짐 = 정수")
    a = ap.parse_args()
    global SEQ
    sfx = "_itw" if a.itw else ""
    dirs = [(d, False) for d in sorted(POSE_SEL.iterdir())]
    if a.itw:
        SEQ = ROOT / "seq_itw"; SEQ.mkdir(exist_ok=True)
        dirs += [(d, True) for d in sorted((ROOT / "pose_sel_itw").iterdir())]
    if a.extra_sel:
        assert a.out, "--extra-sel은 --out과 같이"
        SEQ = ROOT / a.out; SEQ.mkdir(exist_ok=True); sfx = "_" + a.out
        dirs += [(d, "extra") for d in sorted((ROOT / a.extra_sel).iterdir())]
    out = {"train": [], "val": [], "test": [], "itw": []}
    if a.extra_sel:
        out.update(val_extra=[], test_extra=[])
    for ds_dir, extra in dirs:
        if not ds_dir.is_dir():
            continue
        for f in sorted(ds_dir.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            split = str(z["split"]); ds = str(z["dataset"])
            key = split if extra else ("itw" if ds == "OOPS" else split)
            if extra == "extra":
                key = {"train": "train", "val": "val_extra", "test": "test_extra"}[split]
            T = len(z["t"])
            if key in ("train", "val") and T < W:
                continue                                  # 30프레임 미만은 학습에 안 씀
            X = compute_features(z["kpts"], z["conf"], z["box"])
            out[key].append(dict(X=X, y=z["label"].astype(np.int64), y16=z["label16"].astype(np.int64),
                                 dataset=ds, vid=str(z["vid"]), light=str(z["light"]), t=z["t"]))
    for k, lst in out.items():
        with open(SEQ / f"{k}.pkl", "wb") as fh:
            pickle.dump(lst, fh)

    rows = []
    for k, lst in out.items():
        for d in lst:
            for c in (-1, 0, 1, 2):
                rows.append((k, d["dataset"], c, int((d["y"] == c).sum())))
            rows.append((k, d["dataset"], "videos", 1))
    st = pd.DataFrame(rows, columns=["split", "dataset", "label", "n"]).groupby(["split", "dataset", "label"]).n.sum().unstack(fill_value=0)
    st.to_csv(LOGS / f"seq_stats{sfx}.csv")
    print("== 분할·데이터셋별 프레임 수 (-1 제외 / 0 / 1 / 2) 와 영상 수 ==")
    print(st.to_string())

    # 실제 촬영 데이터 비중 맞추기: 학습 조각의 약 20%
    def n_win(d): return max(0, (len(d["X"]) - W) // STEP + 1)
    # --itw: OOPS 학습 조각은 반복 횟수 계산에서 빼서 기존 seq와 같은 반복 횟수를 쓴다 (OOPS는 1번)
    syn = sum(n_win(d) for d in out["train"] if d["dataset"] not in STAGED and d["dataset"] != "OOPS")
    stg = sum(n_win(d) for d in out["train"] if d["dataset"] in STAGED)
    itw_w = sum(n_win(d) for d in out["train"] if d["dataset"] == "OOPS")
    ex_ds = sorted({d["dataset"] for d in out.get("val_extra", []) + out.get("test_extra", [])}) if a.extra_sel else []
    ex_w = sum(n_win(d) for d in out["train"] if d["dataset"] in ex_ds)
    syn -= ex_w                                            # E7: AI Hub 조각은 합성 수에서 뺌 (실제 촬영 반복 횟수는 기존과 같게)
    rep = max(1, int(round(0.25 * syn / stg))) if stg else 1
    share = stg * rep / max(syn + stg * rep + itw_w, 1)
    meta = dict(staged_repeat=rep, syn_windows=syn, staged_windows=stg, staged_share=round(share, 3),
                **({"itw_windows": itw_w} if a.itw else {}))
    if a.extra_sel and ex_w:
        base = syn + stg * rep                             # AI Hub 비중 = ex*R / (base + ex*R) 가 목표에 가장 가깝게
        R = max(1, int(round(a.extra_share / (1 - a.extra_share) * base / ex_w)))
        if a.frac_repeat:
            R = round(a.extra_share / (1 - a.extra_share) * base / ex_w, 2)
        meta.update(extra_repeat={d: R for d in ex_ds}, extra_windows=ex_w, extra_share=round(ex_w * R / (base + ex_w * R), 3))
        print(f"E7: AI Hub 학습 조각 {ex_w}개 x {R}번 반복 -> 학습 조각 중 비중 {meta['extra_share']:.0%} (목표 {a.extra_share:.0%})")
    json.dump(meta, open(SEQ / "meta.json", "w"), indent=1)
    print(f"\n학습 조각: 합성 {syn}개, 실제 촬영 {stg}개 -> 실제 촬영을 {rep}번 반복 (비중 {share:.0%})"
          + (f", OOPS 학습 조각 {itw_w}개" if a.itw else ""))
    append_changes(f"- {'[seq_itw] ' if a.itw else ''}실제 촬영 데이터 반복 횟수 = {rep} (합성 조각 {syn}, 실제 {stg}, 반복 후 비중 {share:.0%})")

    # 특징 상식 확인: 몸통 각도
    tr = out["train"] or out["test"]
    if tr:
        X = np.concatenate([d["X"] for d in tr]); y = np.concatenate([d["y"] for d in tr])
        a0 = X[y == 0, 51].mean() if (y == 0).any() else float("nan")
        a2 = X[y == 2, 51].mean() if (y == 2).any() else float("nan")
        print(f"\n몸통 각도 평균 (0=선 자세, 1=수평): 정상 {a0:.2f}, 쓰러짐 {a2:.2f}  "
              f"{'OK' if a2 > a0 + 0.2 else '!! 차이가 작음 - 관절/정답 확인 필요'}")


if __name__ == "__main__":
    main()
