import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from model import FallGRU
from dataset import NCLS, CLASS_NAMES
from augment import augment

NAMES = "[" + ", ".join(CLASS_NAMES) + "]"

class WinDataset(Dataset):                      # "i번째 샘플을 달라"에 답하는 클래스
    def __init__(self, X, y, aug=False, seed=0):
        self.X, self.y, self.aug = X, y, aug
        self.rng = np.random.default_rng(seed)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = augment(self.X[i], self.rng) if self.aug else self.X[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.long)

def load_split(win_dir, names):
    """split.json의 영상 이름 목록에 해당하는 창만 불러온다"""
    Xs, ys = [], []
    for n in names:
        d = np.load(Path(win_dir) / f"{n}.npz")
        if len(d["y"]):
            Xs.append(d["X"]); ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)

class FocalLoss(nn.Module):
    """가중 Focal Loss: -alpha_y * (1 - p_y)^gamma * log(p_y). 맞히기 쉬운 샘플의 비중을 줄인다."""
    def __init__(self, alpha, gamma=3.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=1).gather(1, y[:, None])[:, 0]
        return (-self.alpha[y] * (1 - logp.exp()) ** self.gamma * logp).mean()

def class_weights(y, cap=30.0):
    counts = np.bincount(y, minlength=NCLS)
    w = len(y) / (NCLS * np.maximum(counts, 1))
    return torch.tensor(np.minimum(w, cap), dtype=torch.float32)

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()

def macro_f1(y, pred):
    return f1_score(y, pred, labels=list(range(NCLS)), average="macro", zero_division=0)

def report(y, pred, title):
    labels = list(range(NCLS))
    print(f"[{title}] 정확도 {float((pred == y).mean()):.3f}  macro-F1 {macro_f1(y, pred):.3f}")
    print(f"클래스별 F1 {NAMES}:",
          np.round(f1_score(y, pred, labels=labels, average=None, zero_division=0), 3).tolist())
    print("혼동행렬 (행=정답, 열=예측):")
    print(confusion_matrix(y, pred, labels=labels))

def run_seed(seed, data, args):
    (X, y), (Xv, yv), (Xt, yt) = data
    torch.manual_seed(seed)
    loader = DataLoader(WinDataset(X, y, aug=not args.no_aug, seed=seed),
                        batch_size=args.batch, shuffle=True)
    model = FallGRU()
    loss_fn = FocalLoss(class_weights(y), gamma=args.gamma)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()                     # 이전 기울기 지우기
            loss = loss_fn(model(xb), yb)       # 예측 -> loss 계산
            loss.backward()                     # 어디를 고칠지 계산
            opt.step()                          # 실제로 고치기
            total += loss.item() * len(yb)
        vf1 = macro_f1(yv, predict(model, Xv))
        if ep == 1 or ep % 10 == 0:
            print(f"  seed {seed} epoch {ep:3d}  loss {total / len(y):.4f}  val macro-F1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  seed {seed} 조기 종료 (epoch {ep}, 최고 val macro-F1 {best_f1:.3f})")
                break
    model.load_state_dict(best_state)
    return model, best_f1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", default="out/win")
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-aug", action="store_true", help="증강 끄기 (증강 유무 비교용)")
    args = ap.parse_args()

    split = json.load(open(args.split))
    data = [load_split(args.win, split[k]) for k in ("train", "val", "test")]
    (X, y), (Xv, yv), (Xt, yt) = data
    print(f"창 수 train {len(y)} / val {len(yv)} / test {len(yt)}  증강 {'끔' if args.no_aug else '켬'}")
    print(f"train 클래스별 개수 {NAMES}:", np.bincount(y, minlength=NCLS).tolist())
    print("클래스 가중치:", np.round(class_weights(y).numpy(), 2).tolist())

    Path(args.out).mkdir(parents=True, exist_ok=True)
    results, best = [], (-1.0, None)
    for s in args.seeds:
        model, vf1 = run_seed(s, data, args)
        tpred = predict(model, Xt)
        tf1 = macro_f1(yt, tpred)
        results.append((vf1, tf1))
        print(f"seed {s}: val macro-F1 {vf1:.3f} / test macro-F1 {tf1:.3f}")
        torch.save(model.state_dict(), Path(args.out) / f"best_seed{s}.pt")
        if vf1 > best[0]:
            best = (vf1, s)
            torch.save(model.state_dict(), Path(args.out) / "best.pt")   # 검증 점수가 제일 높은 시드

    r = np.array(results)
    print(f"\n시드 {len(r)}개 평균±표준편차  val macro-F1 {r[:,0].mean():.3f}±{r[:,0].std():.3f}  "
          f"test macro-F1 {r[:,1].mean():.3f}±{r[:,1].std():.3f}")
    print(f"best.pt = seed {best[1]} (val macro-F1 {best[0]:.3f})")
    model = FallGRU(); model.load_state_dict(torch.load(Path(args.out) / "best.pt"))
    report(yv, predict(model, Xv), "val")
    report(yt, predict(model, Xt), "test (best.pt)")
