"""4단계: 영상마다 주인공 1명 고르기 + 프레임별 정답 붙이기 + 확인 2개.

사용법:
  python 04_select_and_label.py --datasets GMDCSA24 caucafall
  python 04_select_and_label.py                # pose/ 에 있는 데이터 전부

피험자 선택:
  train/val: 넘어짐·쓰러짐 구간(없으면 전체)에서 가장 많이 잡힌 트랙. 1등 < 2등 x 1.5 면 제외(ambiguous)
  한 명만 나오는 데이터셋(OF-Syn, GMDCSA24, CAUCAFall): 애매함 제외를 하지 않고, 고른 트랙이 비어 있는 프레임을
            그 트랙과 거의 겹치지 않는 다른 트랙으로 채운다 (가림·누움 뒤 ID가 바뀐 같은 사람). 정답은 쓰지 않음
  test     : 정답을 보지 않고 가장 오래 잡힌 트랙 (실제 시스템과 같은 조건)
  공통      : 잡힌 프레임 < 50% 면 제외(low_coverage)
결과: pose_sel/<dataset>/<vid>.npz, logs/person_log.csv, logs/label_check/*.png, 화면에 확인 결과
"""
import argparse
import numpy as np
import pandas as pd
from common import ROOT, LABELS, POSE, POSE_SEL, LOGS, FPS, to3

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]


def frame_labels(t, segs):
    lab = np.full(len(t), -1, np.int16)
    for s in segs.itertuples():
        lab[(t >= s.start) & (t < s.end)] = int(s.label)
    return lab


# 영상마다 사람이 1명인 데이터셋. 여러 트랙이 생기면 대부분 같은 사람의 ID 바뀜(가림, 누움) 또는 오검출이다.
SINGLE_PERSON = {"of-syn", "GMDCSA24", "caucafall"}
MAX_OVERLAP = 0.1   # 고른 트랙과 동시에 나온 프레임이 10% 넘는 트랙은 다른 물체로 보고 이어붙이지 않는다


def stitch(present, kp, cf, bx, c):
    """한 명만 나오는 영상: 고른 트랙 c가 비어 있는 프레임을, c와 거의 겹치지 않는 다른 트랙으로 채운다.
    정답을 보지 않고 '잡힌 프레임'만 쓰므로 test에도 같은 규칙을 쓴다. 반환: present(T), kpts, conf, box, 채운 프레임 수"""
    pres = present[c].copy(); k = kp[c].copy(); f = cf[c].copy(); b = bx[c].copy()
    filled = 0
    for j in np.argsort(-present.sum(1)):
        if j == c or not present[j].any():
            continue
        pj = present[j]
        if (pj & pres).sum() > MAX_OVERLAP * pj.sum():
            continue
        fill = pj & ~pres
        k[fill], f[fill], b[fill] = kp[j][fill], cf[j][fill], bx[j][fill]
        pres |= fill
        filled += int(fill.sum())
    return pres, k, f, b, filled


def select(present, label16, split, single=False):
    """present (K,T) bool -> (chosen idx or None, score1, score2, reason, label_aware_choice)"""
    K = present.shape[0]
    if K == 0:
        return None, 0, 0, "no_person", None
    fallmask = np.isin(label16, [1, 2])
    region = fallmask if fallmask.any() else np.ones_like(fallmask)
    s_lab = (present & region).sum(1)
    s_all = present.sum(1)
    lab_choice = int(np.argmax(s_lab))
    if split in ("train", "val"):
        order = np.argsort(-s_lab)
        s1, s2 = int(s_lab[order[0]]), int(s_lab[order[1]]) if K > 1 else 0
        if s1 == 0 or (not single and K > 1 and s1 < 1.5 * s2):
            return int(order[0]), s1, s2, "ambiguous", lab_choice
        return int(order[0]), s1, s2, "", lab_choice
    order = np.argsort(-s_all)
    s1, s2 = int(s_all[order[0]]), int(s_all[order[1]]) if K > 1 else 0
    return int(order[0]), s1, s2, "", lab_choice


def grab_frames(video, times):
    """영상에서 times(초)에 가장 가까운 프레임들 (RGB)"""
    import av
    out = [None] * len(times); best = [1e9] * len(times)
    try:
        with av.open(str(video)) as c:
            t0 = None
            for fr in c.decode(video=0):
                if fr.time is None:
                    continue
                t0 = fr.time if t0 is None else t0
                tt = fr.time - t0
                for j, x in enumerate(times):
                    if abs(tt - x) < best[j]:
                        best[j] = abs(tt - x); out[j] = fr.to_ndarray(format="rgb24")
                if tt > max(times) + 0.5:
                    break
    except Exception:
        pass
    return out


def plot_check(path_png, rec, video_file):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from features import compute_features
    t, y16, y = rec["t"], rec["label16"], rec["label"]
    X = compute_features(rec["kpts"], rec["conf"], rec["box"])
    fall_idx = np.where(y16 == 1)[0]; down_idx = np.where(y16 == 2)[0]
    picks = [max(fall_idx[0] - 8, 0) if len(fall_idx) else 0,
             fall_idx[len(fall_idx) // 2] if len(fall_idx) else len(t) // 2,
             min(down_idx[0] + 15, len(t) - 1) if len(down_idx) else len(t) - 1]
    imgs = grab_frames(ROOT / video_file, [float(t[i]) for i in picks]) if (ROOT / video_file).exists() else [None] * 3
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.step(t, y, where="post", label="label (0 normal / 1 fall / 2 fallen / -1 ignore)", lw=2)
    ax.plot(t, X[:, 51], label="torso angle/90 (0=upright, 1=horizontal)", alpha=0.7)
    ax.plot(t, (rec["conf"] >= 0.3).sum(1) / 17, label="visible joints ratio", alpha=0.5)
    for i in picks:
        ax.axvline(t[i], color="k", ls=":")
    ax.set_xlabel("time (s)"); ax.set_ylim(-1.2, 2.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(path_png.stem)
    for j, (i, im) in enumerate(zip(picks, imgs)):
        a2 = fig.add_subplot(2, 3, 4 + j)
        if im is not None:
            a2.imshow(im)
        k, c = rec["kpts"][i], rec["conf"][i]
        for p, q in EDGES:
            if c[p] >= 0.3 and c[q] >= 0.3:
                a2.plot([k[p, 0], k[q, 0]], [k[p, 1], k[q, 1]], "-", color="lime", lw=2)
        if im is None:
            a2.invert_yaxis()
        a2.set_title(f"t={t[i]:.2f}s  label={y[i]}"); a2.axis("off")
    fig.tight_layout(); fig.savefig(path_png, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--itw", action="store_true",
                    help="E6: labels/videos_itw.csv·segments_itw.csv (OOPS train/val) -> pose_sel_itw/ 에 따로 저장 (기존 pose_sel·로그 안 건드림)")
    ap.add_argument("--tag", default=None,
                    help="E7 등: labels/videos_<tag>.csv·segments_<tag>.csv 사용, 로그 이름 뒤에 _<tag> (기본 없음 = 원래와 같음)")
    ap.add_argument("--pose-dir", default=None, help="관절 폴더 (기본 common.POSE)")
    ap.add_argument("--out-dir", default=None, help="결과 폴더 (기본 common.POSE_SEL)")
    ap.add_argument("--single", nargs="*", default=[], help="SINGLE_PERSON에 더할 데이터셋 이름 (예: aihub_vs)")
    a = ap.parse_args()
    global POSE_SEL, POSE
    sfx = "_itw" if a.itw else ""
    if a.itw:
        POSE_SEL = ROOT / "pose_sel_itw"
    if a.tag:
        sfx = f"_{a.tag}"
    if a.pose_dir:
        POSE = ROOT / a.pose_dir
    if a.out_dir:
        POSE_SEL = ROOT / a.out_dir
    SINGLE_PERSON.update(a.single)
    v = pd.read_csv(LABELS / f"videos{sfx}.csv")
    seg = pd.read_csv(LABELS / f"segments{sfx}.csv")
    seg_by = {k: g for k, g in seg.groupby(["dataset", "path"])}
    dsets = a.datasets or [d.name for d in POSE.iterdir() if d.is_dir()]
    logs, fallen_stats, rows_cnt = [], [], []
    (LOGS / f"label_check{sfx}").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for ds in dsets:
        out = POSE_SEL / ds; out.mkdir(parents=True, exist_ok=True)
        vv = v[v.dataset == ds]
        fall_recs = []
        n_done = 0
        for r in vv.itertuples():
            f = POSE / ds / f"{r.vid}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            t = z["t"]; T = len(t)
            kp, cf, bx = z["kpts"], z["conf"], z["box"]
            present = bx.any(-1) if len(kp) else np.zeros((0, T), bool)
            lab16 = frame_labels(t, seg_by.get((ds, r.path), seg.iloc[:0]))
            single = ds in SINGLE_PERSON
            c, s1, s2, reason, lab_c = select(present, lab16, r.split, single)
            filled = 0
            if c is not None and single:
                pres_c, kc, fc, bc, filled = stitch(present, kp, cf, bx, c)
            elif c is not None:
                pres_c, kc, fc, bc = present[c], kp[c], cf[c], bx[c]
            cov = float(pres_c.mean()) if c is not None and T else 0.0
            if reason == "" and cov < 0.5:
                reason = "low_coverage"
            # 쓰러진 사람 관절 검출률 (제외된 영상 포함, 정직하게)
            n2 = int((lab16 == 2).sum())
            det2 = int(((fc >= 0.3).sum(-1) >= 8)[lab16 == 2].sum()) if (c is not None and n2) else 0
            fallen_stats.append((ds, r.split, n2, det2, int(reason == "")))
            logs.append(dict(dataset=ds, vid=r.vid, split=r.split, n_frames=T, n_tracks=len(kp),
                             chosen=-1 if c is None else c, score1=s1, score2=s2, coverage=round(cov, 3), stitched_frames=filled,
                             excluded=int(reason != ""), reason=reason,
                             test_choice_differs=int(r.split == "test" and c is not None and lab_c != c)))
            n_done += 1
            if reason:
                (out / f"{r.vid}.npz").unlink(missing_ok=True)
                continue
            y = to3(lab16)
            y[~pres_c] = -1                        # 사람이 안 잡힌 프레임은 제외
            rec = dict(t=t, kpts=kc, conf=fc, box=bc, label=y.astype(np.int8), label16=lab16.astype(np.int8),
                       dataset=ds, vid=r.vid, split=r.split, light=r.light)
            np.savez_compressed(out / f"{r.vid}.npz", **rec)
            for sp_lab in (-1, 0, 1, 2):
                rows_cnt.append((ds, r.split, sp_lab, int((y == sp_lab).sum())))
            if (lab16 == 1).any():
                fall_recs.append((rec, r.video_file))
        print(f"{ds}: 처리 {n_done}개")
        if not a.no_plots and fall_recs:
            for j in rng.choice(len(fall_recs), min(3, len(fall_recs)), replace=False):
                rec, vf = fall_recs[j]
                plot_check(LOGS / f"label_check{sfx}" / f"{ds}__{rec['vid']}.png", rec, vf)

    pl = pd.DataFrame(logs)
    pl.to_csv(LOGS / f"person_log{sfx}.csv", index=False)
    print("\n== 제외된 영상 (데이터셋·분할·사유) ==")
    print(pl.groupby(["dataset", "split", "reason"]).size().to_string())
    if (pl.split == "test").any():
        print("\n테스트에서 '정답 기준 선택'과 '가장 오래 잡힌 사람 선택'이 다른 영상 수:",
              int(pl.test_choice_differs.sum()), "/", int((pl.split == "test").sum()))
    cnt = pd.DataFrame(rows_cnt, columns=["dataset", "split", "label", "frames"]).groupby(["dataset", "split", "label"]).frames.sum()
    print("\n== 프레임 수 (label -1 제외 / 0 정상 / 1 넘어지는 중 / 2 쓰러짐) ==")
    print(cnt.unstack(fill_value=0).to_string())
    fdf = pd.DataFrame(fallen_stats, columns=["dataset", "split", "n", "det", "kept"])
    fs = fdf.groupby("dataset")[["n", "det"]].sum()
    fk = fdf[fdf.kept == 1].groupby("dataset")[["n", "det"]].sum()
    print("\n== [확인 1] 쓰러진 사람 관절 검출률 (쓰러짐 프레임 중 관절 8개 이상 잡힌 비율) ==")
    print("   기준: 학습에 쓰는 영상(제외 안 된 영상) 70% 이상. 전체(제외 영상 포함)는 보고용")
    for ds, rr in fs.iterrows():
        if rr.n:
            k = fk.loc[ds] if ds in fk.index else None
            kr = k.det / k.n if (k is not None and k.n) else float("nan")
            print(f"  {ds:10s} 전체 {rr.det / rr.n:6.1%} ({int(rr.det)}/{int(rr.n)})   학습에 쓰는 영상 {kr:6.1%}  "
                  f"{'OK' if kr >= 0.7 else '!! 70% 미만: 멈추고 원인 확인'}")
    print("\n== [확인 2] 라벨 그래프:", LOGS / "label_check", "의 그림을 열어서")
    print("   0->1->2 로 바뀌는 시점(점선 위치 장면)이 실제로 넘어지기 시작/쓰러진 장면과 맞는지 눈으로 확인")


if __name__ == "__main__":
    main()
