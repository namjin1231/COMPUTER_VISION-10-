"""6단계: 학습.

사용법:
  python 06_train.py --model tcnte --seed 0 --exp E2
  python 06_train.py --model gru --seed 0 --exp E1
  python 06_train.py --model tcnte --seed 0 --exp E4 --noaug          # 증강 없음
  python 06_train.py --model tcnte_ref --seed 0 --exp E3              # 레퍼런스 방식 (2클래스, 1초 창, 쓰러진 뒤 제외)
  빠른 시험: --max-epochs 2
결과: runs/<exp>/<이름>/best.pt, config.json, train_log.csv
"""
import argparse, json, pickle, time, csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import SEQ, RUNS, STAGED, W
from models import MODELS, focal_loss, causal_test
from train_utils import get_window, ref_frame_labels, ref_window_label, stream_predict, stream_predict_ref

STEP, REF_WIN, REF_STEP = 5, 15, 3


class WindowDS(Dataset):
    def __init__(self, seqs, index, aug, ref, seed, hn=1.0):
        self.seqs, self.index, self.aug, self.ref, self.seed = seqs, index, aug, ref, seed
        self.hn = hn                                   # B-5: 16클래스 5(lie_down) 프레임 손실 배수 (1 = 원래와 같음)
        self.rng = None
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(self.seed * 1000 + (info.id if info else 0))
        vi, s = self.index[i]
        d = self.seqs[vi]
        Y = np.stack([d["y"], d["y16"]], 1)
        win = REF_WIN if self.ref else W
        Xw, Yw = get_window(d["X"], Y, s, self.rng, self.aug, win)
        if self.ref:
            return Xw, np.int64(ref_window_label(ref_frame_labels(Yw[:, 0], Yw[:, 1])))
        if self.hn != 1.0:
            return Xw, Yw[:, 0].astype(np.int64), np.where(Yw[:, 1] == 5, self.hn, 1.0).astype(np.float32)
        return Xw, Yw[:, 0].astype(np.int64)


def build_index(seqs, ref, repeat, extra_repeat=None):
    win, step = (REF_WIN, REF_STEP) if ref else (W, STEP)
    idx = []
    frng = np.random.default_rng(0)                          # 2차(_r2): 소수 반복(예 0.73)일 때 고정된 일부 조각만 (시드와 무관하게 같은 조각)
    for vi, d in enumerate(seqs):
        T = len(d["X"])
        fr = ref_frame_labels(d["y"], d["y16"]) if ref else None
        for s in range(0, T - win + 1, step):
            if ref:
                if ref_window_label(fr[s:s + win]) < 0:
                    continue
            elif (d["y"][s:s + win] < 0).all():
                continue
            r = repeat if d["dataset"] in STAGED else (extra_repeat or {}).get(d["dataset"], 1)
            if r != int(r):                                  # 정수 반복이면 예전과 완전히 같음
                r = int(r) + int(frng.random() < r - int(r))
            idx += [(vi, s)] * int(r)
    return idx


def boundary_ignore(seqs, k):
    """학습용 사본: 정답이 0·1·2 사이에서 바뀌는 지점(t-1 -> t) 앞뒤 k프레임(t-k .. t+k-1)을 -1로.
    -1과의 경계는 치지 않음. 바뀌는 지점은 원래 정답에서 찾는다. 원본(seqs)은 안 바꿈."""
    out = []
    for d in seqs:
        y = d["y"]
        ch = np.where((y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0))[0] + 1
        if len(ch):
            y2 = y.copy()
            for t in ch:
                y2[max(0, t - k):t + k] = -1
            d = dict(d, y=y2)
        out.append(d)
    return out


def class_alpha(seqs, idx, ref):
    if ref:
        lab = np.array([ref_window_label(ref_frame_labels(seqs[v]["y"], seqs[v]["y16"])[s:s + REF_WIN]) for v, s in idx])
        n = np.array([(lab == 0).sum(), (lab == 1).sum()], float)
    else:
        n = np.zeros(3)
        for v, s in idx:
            y = seqs[v]["y"][s:s + W]
            n += [(y == c).sum() for c in range(3)]
    a = np.minimum(n[0] / np.maximum(n, 1), 30.0); a[0] = 1.0
    return torch.tensor(a, dtype=torch.float32), n


def val_score(model, val, dev, ref):
    from sklearn.metrics import f1_score
    P, Y = [], []
    for d in val:
        if ref:
            p = stream_predict_ref(model, d["X"], dev, REF_WIN)
            fr = ref_frame_labels(d["y"], d["y16"])
            for t in range(REF_WIN - 1, len(d["X"])):
                lab = ref_window_label(fr[t - REF_WIN + 1:t + 1])
                if lab >= 0:
                    P.append(int(p[t] >= 0.5)); Y.append(lab)
        else:
            p = stream_predict(model, d["X"], dev).argmax(1)
            m = d["y"] >= 0
            P.append(p[m]); Y.append(d["y"][m])
    if ref:
        P, Y, labels = np.array(P), np.array(Y), [0, 1]
    else:
        P, Y, labels = np.concatenate(P), np.concatenate(Y), [0, 1, 2]
    f1 = f1_score(Y, P, labels=labels, average=None, zero_division=0)
    return f1, float(f1.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="E2")
    ap.add_argument("--noaug", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cosine", action="store_true", help="학습률 코사인 감소 (기본 꺼짐 = 고정 lr)")
    ap.add_argument("--boundary-ignore", type=int, default=0,
                    help="학습 때만 정답이 0·1·2 사이에서 바뀌는 지점 앞뒤 K프레임을 -1로 (기본 0 = 안 함, 평가는 원래 정답)")
    ap.add_argument("--gamma", type=float, default=3.0, help="Focal loss gamma (기본 3)")
    ap.add_argument("--seq-dir", default=None, help="시퀀스 폴더 (기본: common.SEQ = seq)")
    ap.add_argument("--hn-weight", type=float, default=1.0,
                    help="B-5: 원래 16클래스가 5(lie_down)인 학습 프레임의 손실에 곱할 값 (정답은 그대로 0, alpha도 그대로). 기본 1 = 원래와 같음")
    ap.add_argument("--init", default=None, help="E8a: 이 best.pt에서 출발해 이어서 학습 (기본: 없음 = 처음부터)")
    ap.add_argument("--save-all", action="store_true", help="E8: 에폭마다 ep###.pt 저장 (에폭 선택은 별도 스크립트, 기본 꺼짐)")
    ap.add_argument("--extra-val", default=None, help="E8: 에폭마다 이 시퀀스(예: seq_e8/val_extra.pkl)의 macro F1도 기록 (train_log.csv extra_macro 열)")
    a = ap.parse_args()
    global SEQ
    if a.seq_dir:
        from common import ROOT
        SEQ = ROOT / a.seq_dir

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = a.model == "tcnte_ref"
    name = f"{a.model}{'_noaug' if a.noaug else ''}_seed{a.seed}"
    out = RUNS / a.exp / name; out.mkdir(parents=True, exist_ok=True)

    train = pickle.load(open(SEQ / "train.pkl", "rb"))
    val = pickle.load(open(SEQ / "val.pkl", "rb"))
    meta = json.load(open(SEQ / "meta.json"))
    if a.boundary_ignore > 0:
        train = boundary_ignore(train, a.boundary_ignore)
    idx = build_index(train, ref, meta["staged_repeat"], meta.get("extra_repeat"))       # extra_repeat: E7 seq_e7에만 있음
    alpha, n = class_alpha(train, idx, ref)
    print(f"[{name}] device={dev} 학습 조각 {len(idx)}개, 클래스별 수 {n.astype(int).tolist()}, alpha {alpha.numpy().round(2).tolist()}")

    model = MODELS[a.model]().to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev)); print(f"  출발 가중치: {a.init}")
    xval = pickle.load(open(a.extra_val, "rb")) if a.extra_val else None
    if not ref:
        d1, d2 = causal_test(model.cpu()); model.to(dev)
        print(f"  미래 차단 테스트: 미래 변경 영향 {d1:.1e}, 앞부분 일치 {d2:.1e}", "OK" if max(d1, d2) < 1e-5 else "!! 실패")
        if max(d1, d2) >= 1e-5:
            raise SystemExit("미래 차단 테스트 실패 - 멈춤")

    if a.hn_weight != 1.0 and ref:
        raise SystemExit("--hn-weight는 프레임 단위 모델에만")
    ds = WindowDS(train, idx, not a.noaug, ref, a.seed, a.hn_weight)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0, pin_memory=dev == "cuda")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_epochs) if a.cosine else None
    alpha = alpha.to(dev)
    best, bad = -1.0, 0
    log = open(out / "train_log.csv", "w", newline="")
    lw = csv.writer(log); lw.writerow(["epoch", "loss", "val_f1", "val_macro", "secs"] + (["extra_f1", "extra_macro"] if xval is not None else []))
    t_all = time.time()
    for ep in range(a.max_epochs):
        t0 = time.time(); model.train(); tot = nb = 0
        for batch in dl:
            X, Y = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
            out_ = model(X)
            if a.hn_weight != 1.0:
                loss = focal_loss(out_.reshape(-1, out_.shape[-1]), Y.reshape(-1), alpha, a.gamma, batch[2].to(dev).reshape(-1))
            else:
                loss = focal_loss(out_.reshape(-1, out_.shape[-1]), Y.reshape(-1), alpha, a.gamma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if sched is not None:
            sched.step()
        f1, mf = val_score(model, val, dev, ref)
        xr = []
        if xval is not None:
            xf1, xmf = val_score(model, xval, dev, ref); xr = [xf1.round(4).tolist(), round(xmf, 4)]
            print(f"  ep{ep:03d} extra val F1 {np.round(xf1, 3).tolist()} macro {xmf:.3f}", flush=True)
        if a.save_all:
            torch.save(model.state_dict(), out / f"ep{ep:03d}.pt")
        secs = time.time() - t0
        lw.writerow([ep, round(tot / max(nb, 1), 5), f1.round(4).tolist(), round(mf, 4), round(secs, 1)] + xr); log.flush()
        print(f"  ep{ep:03d} loss {tot / max(nb, 1):.4f}  val F1 {np.round(f1, 3).tolist()} macro {mf:.3f}  {secs:.0f}초", flush=True)
        if ep == 0:
            print(f"  -> 1에폭 {secs:.0f}초. 최대 {a.max_epochs}에폭이면 최대 약 {secs * a.max_epochs / 60:.0f}분 (보통 조기 종료로 더 짧음)")
        if mf > best:
            best, bad = mf, 0
            torch.save(model.state_dict(), out / "best.pt")
            json.dump(dict(model=a.model, seed=a.seed, exp=a.exp, noaug=a.noaug, ref=ref, epoch=ep, val_macro=mf,
                           val_f1=f1.tolist(), alpha=alpha.cpu().tolist(), lr=a.lr, n_windows=len(idx),
                           staged_repeat=meta["staged_repeat"], **({"cosine": True} if a.cosine else {}),
                           **({"boundary_ignore": a.boundary_ignore} if a.boundary_ignore else {}),
                           **({"gamma": a.gamma} if a.gamma != 3.0 else {}),
                           **({"seq_dir": a.seq_dir} if a.seq_dir else {}),
                           **({"hn_weight": a.hn_weight} if a.hn_weight != 1.0 else {}),
                           **({"init": a.init} if a.init else {})),
                      open(out / "config.json", "w"), indent=1)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  조기 종료 (검증 점수가 {a.patience}에폭 동안 안 오름)")
                break
    log.close()
    print(f"[{name}] 끝: 최고 val macro-F1 {best:.3f}, 총 {(time.time() - t_all) / 60:.0f}분 -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
