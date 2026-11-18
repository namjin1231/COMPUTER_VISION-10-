"""④ 분류기 학습 (제안서 4.3 ④)

  python train.py --datasets GMDCSA24 --model gru --out out/runs/e1_gru                 # E1
  python train.py --datasets of-syn GMDCSA24 caucafall --model tcnte --out out/runs/e2_tcnte

  손실      가중 Focal Loss (γ = 3, 클래스 가중치 최대 30). 프레임별 정답, 제외 프레임은 손실에서 뺌
  최적화    Adam lr 1e-3, 배치 64, 최대 100 epoch
  모델 선택 검증 세트를 실시간 방식으로 예측한 프레임별 macro-F1. 조기 종료
  반복      시드 3개. 시드별 seed<N>.pt, 검증 점수가 가장 높은 시드를 best.pt로
  실제 영상 합성 데이터와 같이 학습할 때 실제 촬영 창을 반복해 한 epoch의 약 20%가 되게 함
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import augment_batch, sample_speed
from dataset import CLASS_NAMES, EXCLUDE, NCLS, SYNTHETIC, Corpus, gather
from evaluate import macro_f1
from infer import predict_corpus
from models import build_model, n_params, save_ckpt


def focal_loss(logits, y, alpha, gamma):
    """logits (B, T, 3), y (B, T) 제외는 -1. -alpha_y (1 - p_y)^gamma log p_y 의 평균"""
    keep = y != EXCLUDE
    logp = F.log_softmax(logits[keep], -1).gather(1, y[keep][:, None])[:, 0]
    return (-alpha[y[keep]] * (1 - logp.exp()) ** gamma * logp).mean()


def class_weights(corpus, vid, end, cap=30.0):
    """학습 창에 들어가는 프레임 수 기준. 적은 클래스일수록 크게, 최대 cap"""
    _, Y = gather(corpus, vid, end)
    cnt = np.array([(Y == c).sum() for c in range(NCLS)], float)
    return np.minimum(cnt.sum() / (NCLS * np.maximum(cnt, 1)), cap), cnt.astype(int)


def epoch_order(corpus, vid, rng, real_frac):
    """한 epoch에 쓸 창 순서. 합성과 실제가 섞여 있고 실제 비중이 real_frac보다 낮으면 실제 창을 반복한다."""
    n = len(vid)
    real = np.array([d not in SYNTHETIC for d in corpus.dataset])[vid]
    if real.all() or not real.any() or real.mean() >= real_frac:
        return rng.permutation(n)
    p = np.where(real, real_frac / real.sum(), (1 - real_frac) / (~real).sum())
    return rng.choice(n, size=n, replace=True, p=p)


def run_seed(seed, train, val, a, out):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vid, end = train.windows()
    w, cnt = class_weights(train, vid, end)
    alpha = torch.tensor(w, dtype=torch.float32)
    model = build_model(a.model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_ep, bad, hist = -1.0, 0, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        order = epoch_order(train, vid, rng, a.real_frac)
        total = 0.0
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            if a.no_aug:
                X, Y = gather(train, vid[b], end[b])
            else:
                sp = sample_speed(len(b), rng)
                X, Y = gather(train, vid[b], end[b], sp)
                X = augment_batch(X, sp, rng)
            opt.zero_grad()
            loss = focal_loss(model(torch.from_numpy(X)), torch.from_numpy(Y), alpha, a.gamma)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        vf1 = macro_f1(val.y, predict_corpus(model, val).argmax(1))
        hist.append(dict(epoch=ep, loss=total / len(order), val_f1=vf1))
        if vf1 > best:
            best, best_ep, bad = vf1, ep, 0
            save_ckpt(out / f"seed{seed}.pt", model, a.model, seed=seed, epoch=ep, val_macro_f1=vf1,
                      datasets=a.datasets, args=vars(a))
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0 or bad >= a.patience:
            print(f"  seed {seed} epoch {ep:3d}  loss {total / len(order):.4f}  val macro-F1 {vf1:.3f}  "
                  f"(최고 {best:.3f} @ {best_ep})  {time.time() - t0:.0f}s", flush=True)
        if bad >= a.patience:
            break
    return dict(seed=seed, val_macro_f1=best, epoch=best_ep, history=hist,
                class_weights=w.round(2).tolist(), class_frames=cnt.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/data")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--model", default="tcnte", choices=["tcnte", "gru", "cnn"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--real-frac", type=float, default=0.2)
    ap.add_argument("--no-aug", action="store_true", help="증강 끄기 (증강 유무 비교용)")
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU 스레드 수. 모델이 작아 1이 가장 빠르다 (TCNTE 실측: 8스레드는 1스레드보다 4배 느림)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val = Corpus(a.data, a.datasets, "train"), Corpus(a.data, a.datasets, "val")
    vid, end = train.windows()
    w, cnt = class_weights(train, vid, end)
    print(f"모델 {a.model} (파라미터 {n_params(build_model(a.model)):,}개), 데이터 {a.datasets}, "
          f"증강 {'끔' if a.no_aug else '켬'}")
    print(f"학습 영상 {len(train)}개, 창 {len(vid)}개 / 검증 영상 {len(val)}개, 프레임 {int((val.y != EXCLUDE).sum())}")
    print(f"학습 프레임 {CLASS_NAMES} = {cnt.tolist()}, 클래스 가중치 {w.round(2).tolist()}")

    results = [run_seed(s, train, val, a, out) for s in a.seeds]
    f1 = np.array([r["val_macro_f1"] for r in results])
    top = results[int(f1.argmax())]
    (out / "best.pt").write_bytes((out / f"seed{top['seed']}.pt").read_bytes())
    print(f"검증 macro-F1 {f1.mean():.3f} ± {f1.std():.3f} "
          f"{[round(float(v), 3) for v in f1]}, best.pt = seed {top['seed']} (epoch {top['epoch']})")
    json.dump(dict(model=a.model, datasets=a.datasets, args=vars(a), val_macro_f1_mean=float(f1.mean()),
                   val_macro_f1_std=float(f1.std()), best_seed=top["seed"], seeds=results),
              open(out / "summary.json", "w"), ensure_ascii=False, indent=1)
