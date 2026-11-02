import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from model import FallGRU

class WinDataset(Dataset):                      # "i번째 샘플을 달라"에 답하는 클래스
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def load_all(win_dir):
    Xs, ys = [], []
    for p in sorted(Path(win_dir).glob("*.npz")):
        d = np.load(p)
        Xs.append(d["X"]); ys.append(d["y"])
        print(p.name, d["X"].shape)
    return np.concatenate(Xs), np.concatenate(ys)

if __name__ == "__main__":
    torch.manual_seed(0)
    X, y = load_all(sys.argv[1])
    counts = np.bincount(y, minlength=3)
    print("클래스별 개수 [정상, 넘어지는 중, 쓰러짐]:", counts.tolist())

    w = torch.tensor(len(y) / (3 * np.maximum(counts, 1)), dtype=torch.float32)  # 적은 클래스에 가중치
    loader = DataLoader(WinDataset(X, y), batch_size=8, shuffle=True)            # 8개씩 묶어서 공급
    model = FallGRU()
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for ep in range(1, 301):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()                     # 이전 기울기 지우기
            loss = loss_fn(model(xb), yb)       # 예측 -> loss 계산
            loss.backward()                     # 어디를 고칠지 계산
            opt.step()                          # 실제로 고치기
            total += loss.item() * len(yb)
        if ep == 1 or ep % 20 == 0:
            print(f"epoch {ep:3d}  loss {total / len(y):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    print("정확도:", float((pred == y).mean()))
    print("혼동행렬 (행=정답, 열=예측):")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))