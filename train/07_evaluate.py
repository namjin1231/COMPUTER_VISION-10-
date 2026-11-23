"""7단계: 평가 (실시간 방식). 판정 임계값은 검증(val) 세트로만 정하고, 테스트는 마지막에 한 번 쓴다.

사용법:
  python 07_evaluate.py --runs runs/E2/tcnte_seed0
  python 07_evaluate.py --runs runs/E2/*              # 여러 개 (윈도우 cmd에서도 * 동작)
결과: results.csv (experiment, model, seed, subset, metric, value), runs/.../eval.json, 화면 요약

지표
  frame_*    : 프레임별 정밀도·재현율·F1, macro-F1 (정답 -1 제외)
  seg_*      : 구간 단위 (구간 안 프레임 예측의 다수결). OmniFall 논문 비교용. lying 구간은 음성으로 셈
  vid_*      : 영상 단위 민감도/특이도. logic=판정 로직(넘어짐->쓰러짐 확인), model=P(쓰러짐)만, rule=규칙만,
               fallonly=넘어지는 순간 감지만 (레퍼런스 방식과 같은 기준)
  fa_per_hour: 일상 영상에서 시간당 '쓰러짐 확인' 횟수,  delay_s: 넘어짐 시작 -> 확인까지 초
"""
import argparse, glob, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ROOT, SEQ, LABELS, FPS
from models import MODELS
from train_utils import stream_predict, stream_predict_ref, confirm_frame, runs_of, speed_max

REF_WIN = 15


# ---------- 탐지 방식들 ----------
VGATES = (None,)     # B-3 --v-gate-grid: spec95 격자에 v_gate 후보 추가 (기본 = 끔만 = 원래와 같음)


def det_logic(prob, vmax, th):
    """vmax: 영상의 speed_max(X) (v_gate 쓸 때만 필요, 안 쓰면 None)"""
    return confirm_frame(prob, th_fall=th["th_fall"], th_down=th["th_down"], n_down=th.get("n_down", 8),
                         window_s=th.get("window_s", 3.0), vmax=vmax, v_gate=th.get("v_gate"))


def smooth_probs(p, n):
    """판정 로직용: 각 프레임 확률을 '그 프레임까지 최근 n프레임' 평균으로 (과거만 사용). n<=1이면 그대로"""
    if n <= 1:
        return p
    c = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), 0)
    i = np.arange(1, len(p) + 1); lo = np.maximum(i - n, 0)
    return ((c[i] - c[lo]) / (i - lo)[:, None]).astype(np.float32)


GRID_TF, GRID_TD, GRID_ND = (0.3, 0.4, 0.5, 0.6, 0.7), (0.4, 0.5, 0.6, 0.7, 0.8), (4, 8, 12)


def logic_grid(val, sprobs):
    """판정 로직 후보 전부의 val (민감도, 특이도, th_fall, th_down, n_down)"""
    fall, adl = video_sets(val)
    out = []
    vms = [speed_max(d["X"]) for d in val] if VGATES != (None,) else [None] * len(val)
    for vg in VGATES:                              # 끔(None)이 먼저 -> 동점이면 끔
        for nd in GRID_ND:
            for tf in GRID_TF:
                for td in GRID_TD:
                    t = {"th_fall": tf, "th_down": td, "n_down": nd}
                    if vg is not None:
                        t["v_gate"] = vg
                    se, sp = sens_spec_state([det_logic(p, vm, t) for p, vm in zip(sprobs, vms)], val, fall, adl)
                    out.append((se, sp, tf, td, nd) if VGATES == (None,) else (se, sp, tf, td, nd, vg))
    return out


def sens_at_spec(grid, spec=0.95):
    """val 특이도 >= spec 인 후보 중 최대 민감도 (없으면 nan)"""
    ok = [se for se, sp, *_ in grid if sp >= spec]
    return max(ok) if ok else np.nan
def det_model(prob, X, th):
    c = np.where(runs_of(prob[:, 2] >= 0.6, 8))[0]; return int(c[0]) if len(c) else None
def det_rule(prob, X, th):
    ang = np.where(runs_of(X[:, 51] > 0.667, 15))[0]
    for t in ang:
        s = t - 14
        if X[max(0, s - 15):s + 1, 54].max() > th["v_th"]:
            return int(t)
    return None
def fall_moments(prob, th):     return runs_of(prob[:, 1] >= th["th_fall"], 2)
def fall_moments_ref(p, th):    return np.nan_to_num(p, nan=0.0) >= th["th_ref"]


def video_sets(seqs):
    fall = [i for i, d in enumerate(seqs) if (d["y16"] == 1).any()]
    adl = [i for i, d in enumerate(seqs) if not np.isin(d["y16"], [1, 2]).any()]
    return fall, adl


def sens_spec_state(dets, seqs, fall, adl):
    tp = sum(dets[i] is not None for i in fall); fp = sum(dets[i] is not None for i in adl)
    se = tp / len(fall) if fall else np.nan; sp = 1 - fp / len(adl) if adl else np.nan
    return se, sp


def sens_spec_moment(moms, seqs, fall, adl):
    tp = 0
    for i in fall:
        fr = np.where(seqs[i]["y16"] == 1)[0]
        tp += bool(moms[i][fr[0]:fr[-1] + 16].any())
    fp = sum(bool(moms[i].any()) for i in adl)
    return (tp / len(fall) if fall else np.nan), (1 - fp / len(adl) if adl else np.nan)


def bal(se, sp):
    return np.nanmean([se, sp])


def tune(val, probs, ref, objective="bal", sprobs=None):
    """objective: bal(기본, 지금 방식) / spec95 (val 특이도 0.95 이상 중 민감도 최대, n_down도 고름).
    sprobs: 판정 로직용 확률 (smooth 적용본). None이면 probs 그대로"""
    fall, adl = video_sets(val)
    th = {}
    if ref:
        best = max(((bal(*sens_spec_moment([fall_moments_ref(p, {"th_ref": t}) for p in probs], val, fall, adl)), t)
                    for t in (0.3, 0.4, 0.5, 0.6, 0.7)))
        return {"th_ref": best[1]}
    sprobs = probs if sprobs is None else sprobs
    if objective == "spec95":
        g = logic_grid(val, sprobs)
        # 특이도 0.95 이상이면 민감도 최대(동점은 특이도 높은 쪽), 하나도 없으면 특이도 최대
        best = max(g, key=lambda r: (r[1] >= 0.95, r[0] if r[1] >= 0.95 else r[1], r[1]))
        th["th_fall"], th["th_down"], th["n_down"] = best[2], best[3], best[4]
        if VGATES != (None,):
            th["v_gate"] = best[5]                 # None = 끔
    else:
        cand = []
        for tf in (0.3, 0.4, 0.5, 0.6, 0.7):
            for td in (0.4, 0.5, 0.6, 0.7, 0.8):
                t = {"th_fall": tf, "th_down": td}
                se, sp = sens_spec_state([det_logic(p, None, t) for p in sprobs], val, fall, adl)
                cand.append((bal(se, sp), sp, tf, td))
        _, _, th["th_fall"], th["th_down"] = max(cand)
    cand = []
    for v in np.arange(0.01, 0.21, 0.01):
        t = {"v_th": float(v)}
        se, sp = sens_spec_state([det_rule(None, d["X"], t) for d in val], val, fall, adl)
        cand.append((bal(se, sp), sp, float(v)))
    th["v_th"] = max(cand)[2]
    return th


def frame_metrics(P, Y, labels):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p, r, f, _ = precision_recall_fscore_support(Y, P, labels=labels, zero_division=0)
    return p, r, f, confusion_matrix(Y, P, labels=labels)


def merge_results():
    """runs/*/*/results.csv 를 모아 results.csv 로"""
    from common import RUNS, RESULTS              # RESULTS = ROOT/results.csv (FALL_VARIANT 있으면 results_<값>.csv)
    fs = sorted(RUNS.glob("*/*/results.csv"))
    if fs:
        pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(RESULTS, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["test", "itw"])
    ap.add_argument("--ensemble", action="store_true", help="같은 실험·모델의 시드들 확률 평균 -> <모델>_ens 폴더에 저장")
    ap.add_argument("--smooth", type=int, default=1, help="판정 로직 전에 최근 N프레임 평균 (과거만, 1 = 안 함)")
    ap.add_argument("--objective", choices=["bal", "spec95"], default="bal",
                    help="val 임계값 고르기: bal(기본) / spec95(특이도 0.95 이상 중 민감도 최대, n_down 4/8/12도)")
    ap.add_argument("--report-val", action="store_true", help="val 영상 단위 지표(선택점, 특이도0.95일 때 민감도)도 결과에 넣기")
    ap.add_argument("--tag", default="", help="결과를 results_<tag>.csv / eval_<tag>.json 로 따로 저장 (전체 결과표는 안 건드림)")
    ap.add_argument("--seq-dir", default=None, help="시퀀스 폴더 (기본: common.SEQ = seq)")
    ap.add_argument("--v-gate-grid", nargs="*", type=float, default=None,
                    help="B-3: spec95 격자에 v_gate 후보 추가 (끔 포함). 값 없이 쓰면 0.02 0.03 0.05 0.08 0.12")
    ap.add_argument("--labels-tag", default="", help="labels/videos_<tag>.csv·segments_<tag>.csv 사용 (내 데이터셋. 기본 = videos.csv·segments.csv)")
    ap.add_argument("--report-liedown", action="store_true", help="--report-val에 '눕기 영상 오경보율' 행 추가")
    a = ap.parse_args()
    global SEQ, VGATES
    if a.v_gate_grid is not None:
        VGATES = (None,) + tuple(a.v_gate_grid or (0.02, 0.03, 0.05, 0.08, 0.12))
    if a.seq_dir:
        SEQ = ROOT / a.seq_dir
    runs = [Path(p) for pat in a.runs for p in sorted(glob.glob(pat)) if (Path(p) / "best.pt").exists()]
    if not runs:
        raise SystemExit("best.pt 있는 run 폴더가 없음")
    sfx = f"_{a.tag}" if a.tag else ""
    # 평가 단위: 기본은 run 하나씩, --ensemble 이면 (실험, 모델, noaug, ref)가 같은 run들을 묶음
    units = []
    if a.ensemble:
        grp = {}
        for rd in runs:
            c = json.load(open(rd / "config.json"))
            grp.setdefault((c["exp"], c["model"], bool(c.get("noaug")), bool(c.get("ref"))), []).append(rd)
        for (exp, mdl, noaug, _), rds in grp.items():
            c = json.load(open(rds[0] / "config.json"))
            name = mdl + ("_noaug" if noaug else "") + "_ens"
            out = rds[0].parent / name; out.mkdir(exist_ok=True)
            c = dict(c, seed="ens", members=[r.name for r in rds])
            json.dump(c, open(out / "config.json", "w"), indent=1)
            units.append((out, c, rds))
    else:
        units = [(rd, None, [rd]) for rd in runs]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = pickle.load(open(SEQ / "val.pkl", "rb"))
    sets = {k: pickle.load(open(SEQ / f"{k}.pkl", "rb")) for k in a.sets if (SEQ / f"{k}.pkl").exists()}
    lt = f"_{a.labels_tag}" if a.labels_tag else ""
    vids = pd.read_csv(LABELS / f"videos{lt}.csv")
    path_of = {(r.dataset, r.vid): r.path for r in vids.itertuples()}
    seg = pd.read_csv(LABELS / f"segments{lt}.csv")
    seg_by = {k: g for k, g in seg.groupby(["dataset", "path"])}

    for rd, ens_cfg, members in units:
        cfg = ens_cfg or json.load(open(rd / "config.json"))
        ref = cfg.get("ref", False)
        mname = cfg["model"] + ("_noaug" if cfg.get("noaug") else "") + ("_ens" if ens_cfg else "")
        models = []
        for m_rd in members:
            mdl = MODELS[cfg["model"]]().to(dev)
            mdl.load_state_dict(torch.load(m_rd / "best.pt", map_location=dev))
            models.append(mdl)
        def pred1(model, X):
            return stream_predict_ref(model, X, dev, REF_WIN) if ref else stream_predict(model, X, dev)
        if len(models) == 1:                                   # 기본: 원래와 똑같은 계산
            pred = lambda X, m=models[0]: pred1(m, X)
        else:                                                  # 앙상블: 시드별 확률 평균
            pred = lambda X: np.mean(np.stack([pred1(m, X) for m in models]), 0).astype(np.float32)
        vprobs = [pred(d["X"]) for d in val]
        vsprobs = [smooth_probs(p, a.smooth) for p in vprobs] if not ref else vprobs
        th = tune(val, vprobs, ref, a.objective, vsprobs)
        rows = [("val", f"th_{k}" if not k.startswith("th") else k, (-1 if v is None else v)) for k, v in th.items()]   # v_gate 끔 = -1
        use_vg = th.get("v_gate") is not None
        vvm = [speed_max(d["X"]) for d in val] if use_vg else [None] * len(val)
        if a.report_val and not ref:
            vf, va = video_sets(val)
            vdl = [det_logic(p, vm, th) for p, vm in zip(vsprobs, vvm)]
            se, sp = sens_spec_state(vdl, val, vf, va)
            rows += [("val", "vid_logic_sens", se), ("val", "vid_logic_spec", sp),
                     ("val", "vid_logic_sens_at_spec95", sens_at_spec(logic_grid(val, vsprobs), 0.95)),
                     ("val", "smooth", a.smooth), ("val", "objective", a.objective)]
            Pv = np.concatenate([p.argmax(1)[d["y"] >= 0] for p, d in zip(vprobs, val)])
            Yv = np.concatenate([d["y"][d["y"] >= 0] for d in val])
            rows += [("val", "val_frame_macro_f1", float(frame_metrics(Pv, Yv, [0, 1, 2])[2].mean()))]
            if a.report_liedown:                   # 눕기 영상 오경보율: 정상 영상 중 y16에 5(lie_down)가 있는 영상에서 경보 비율
                ld = [i for i in va if (val[i]["y16"] == 5).any()]
                rows += [("val", "vid_logic_liedown_fa", sum(vdl[i] is not None for i in ld) / len(ld) if ld else np.nan),
                         ("val", "n_liedown_videos", len(ld))]
        print(f"\n===== {rd}  (val로 정한 임계값: {th}{', smooth ' + str(a.smooth) if a.smooth > 1 else ''}) =====")

        for sname, seqs in sets.items():
            probs = [pred(d["X"]) for d in seqs]
            sprobs = [smooth_probs(p, a.smooth) for p in probs] if not ref else probs   # 판정 로직(logic)에만 씀
            groups = {}
            for i, d in enumerate(seqs):
                groups.setdefault(d["dataset"], []).append(i)
                if d["light"] in ("day", "dark"):
                    groups.setdefault(f"{d['dataset']}_{d['light']}", []).append(i)
            if sname == "test":
                groups["test_all"] = list(range(len(seqs)))
            for g, ids in groups.items():
                sub = [seqs[i] for i in ids]; sp_ = [probs[i] for i in ids]; ssp_ = [sprobs[i] for i in ids]
                fall, adl = video_sets(sub)
                rows += [(g, "n_videos", len(sub)), (g, "n_fall_videos", len(fall)), (g, "n_adl_videos", len(adl))]
                if ref:
                    se, sp = sens_spec_moment([fall_moments_ref(p, th) for p in sp_], sub, fall, adl)
                    rows += [(g, "vid_fallonly_sens", se), (g, "vid_fallonly_spec", sp)]
                    continue
                # 프레임 단위
                P = np.concatenate([p.argmax(1)[d["y"] >= 0] for p, d in zip(sp_, sub)])
                Y = np.concatenate([d["y"][d["y"] >= 0] for d in sub])
                if len(Y):
                    pr, rc, f1, cm = frame_metrics(P, Y, [0, 1, 2])
                    for c, nm in enumerate(["normal", "fall", "fallen"]):
                        rows += [(g, f"frame_{nm}_p", pr[c]), (g, f"frame_{nm}_r", rc[c]), (g, f"frame_{nm}_f1", f1[c])]
                    rows += [(g, "frame_macro_f1", f1.mean()), (g, "frame_confusion", json.dumps(cm.tolist()))]
                # 영상 단위
                dl = [det_logic(p, speed_max(d["X"]) if use_vg else None, th) for p, d in zip(ssp_, sub)]
                for nm, dets in [("logic", dl), ("model", [det_model(p, None, th) for p in sp_]),
                                 ("rule", [det_rule(None, d["X"], th) for d in sub])]:
                    se, sp = sens_spec_state(dets, sub, fall, adl)
                    rows += [(g, f"vid_{nm}_sens", se), (g, f"vid_{nm}_spec", sp)]
                se, sp = sens_spec_moment([fall_moments(p, th) for p in sp_], sub, fall, adl)
                rows += [(g, "vid_fallonly_sens", se), (g, "vid_fallonly_spec", sp)]
                delays = [(dl[i] - np.where(sub[i]["y16"] == 1)[0][0]) / FPS for i in fall if dl[i] is not None]
                hours = sum(len(sub[i]["y"]) for i in adl) / FPS / 3600
                rows += [(g, "delay_s", float(np.mean(delays)) if delays else np.nan),
                         (g, "fa_per_hour", sum(dl[i] is not None for i in adl) / hours if hours else np.nan)]
                # 구간 단위 (OmniFall 비교용)
                tp = {1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}          # TP FP FN TN
                for p, d in zip(sp_, sub):
                    sg = seg_by.get((d["dataset"], path_of.get((d["dataset"], d["vid"]), "")))
                    if sg is None:
                        continue
                    am = p.argmax(1)
                    for s in sg.itertuples():
                        m = (d["t"] >= s.start) & (d["t"] < s.end)
                        if not m.any():
                            continue
                        pc = int(np.bincount(am[m], minlength=3).argmax())
                        gc = 1 if s.label == 1 else 2 if s.label == 2 else 0
                        for c in (1, 2):
                            k = 0 if (pc == c and gc == c) else 1 if pc == c else 2 if gc == c else 3
                            tp[c][k] += 1
                for c, nm in ((1, "fall"), (2, "fallen")):
                    TP, FP, FN, TN = tp[c]
                    if TP + FN:
                        rows += [(g, f"seg_{nm}_sens", TP / (TP + FN)), (g, f"seg_{nm}_spec", TN / max(TN + FP, 1)),
                                 (g, f"seg_{nm}_f1", 2 * TP / max(2 * TP + FP + FN, 1))]
                # 누운 구간 참고
                ly = np.concatenate([p.argmax(1)[d["y16"] == 6] for p, d in zip(sp_, sub)])
                if len(ly):
                    bc = np.bincount(ly, minlength=3) / len(ly)
                    rows += [(g, "lying_as_normal", bc[0]), (g, "lying_as_fall", bc[1]), (g, "lying_as_fallen", bc[2])]

        df = pd.DataFrame(rows, columns=["subset", "metric", "value"])
        df.insert(0, "seed", cfg["seed"]); df.insert(0, "model", mname); df.insert(0, "experiment", cfg["exp"])
        df.to_csv(rd / f"results{sfx}.csv", index=False)      # run 폴더마다 따로 저장 (동시에 돌려도 안전)
        if not a.tag:                                          # --tag 시험 결과는 전체 결과표에 안 넣음
            merge_results()
        show = df
        key = ["frame_macro_f1", "frame_fall_f1", "frame_fallen_f1", "vid_logic_sens", "vid_logic_spec",
               "vid_fallonly_sens", "vid_fallonly_spec", "seg_fall_f1", "seg_fallen_f1", "fa_per_hour", "n_videos"]
        tab = show[show.metric.isin(key)].assign(value=lambda x: pd.to_numeric(x.value, errors="coerce")).pivot_table(index="subset", columns="metric", values="value", aggfunc="first")
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
            print(tab.reindex(columns=[k for k in key if k in tab.columns]).to_string())
        json.dump(dict(thresholds=th) if not (a.tag or a.smooth > 1 or a.objective != "bal") else
                  dict(thresholds=th, smooth=a.smooth, objective=a.objective), open(rd / f"eval{sfx}.json", "w"), indent=1)
    from common import RESULTS
    print("\n저장:", RESULTS if not a.tag else f"각 run 폴더의 results{sfx}.csv")


if __name__ == "__main__":
    main()
