"""models.py: TCNTE(메인), GRU, 1D-CNN. 입력 (B,T,55) -> 출력 (B,T,3) 프레임별 logit. 셋 다 미래 프레임을 보지 않는다.
TCNTERef: E3용 레퍼런스 방식 (미래 차단 없음, 시간축 평균, 창마다 2클래스)."""
import torch, torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

W = 30


class Chomp(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return x[:, :, :-self.n] if self.n else x


class TBlock(nn.Module):
    def __init__(self, i, o, k, d, p=0.2):
        super().__init__()
        pad = (k - 1) * d                                  # 왼쪽만 남기는 인과 합성곱
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d)), Chomp(pad), nn.ReLU(), nn.Dropout(p),
            weight_norm(nn.Conv1d(o, o, k, padding=pad, dilation=d)), Chomp(pad), nn.ReLU(), nn.Dropout(p))
        self.down = nn.Conv1d(i, o, 1) if i != o else None
    def forward(self, x):
        r = x if self.down is None else self.down(x)
        return torch.relu(self.net(x) + r)


class TCNTE(nn.Module):
    # 기본값(ch 8/16/32, d 32, 헤드 4, FF 32)이면 원래 TCNTE와 완전히 같음 (모듈 생성 순서·초기화·키 이름 동일)
    def __init__(self, nin=55, ncls=3, ch=(8, 16, 32), d=32, heads=4, ff=32):
        super().__init__()
        self.tcn = nn.Sequential(TBlock(nin, ch[0], 5, 1), TBlock(ch[0], ch[1], 5, 2), TBlock(ch[1], ch[2], 5, 4))
        self.pos = nn.Parameter(torch.zeros(1, W, d))
        nn.init.normal_(self.pos, std=0.02)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, 0.1, batch_first=True), 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, ncls))
    def forward(self, x):                                  # (B,T,55) -> (B,T,3) logits, T <= 30
        T = x.shape[1]
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)   # 미래 차단
        return self.head(self.te(h, mask=mask))


class GRUNet(nn.Module):
    def __init__(self, nin=55, ncls=3):
        super().__init__()
        self.inp = nn.Linear(nin, 128)
        self.gru = nn.GRU(128, 128, 2, batch_first=True, dropout=0.3)
        self.out = nn.Linear(128, ncls)
    def forward(self, x):
        h, _ = self.gru(torch.relu(self.inp(x)))
        return self.out(h)


class CConv(nn.Module):
    def __init__(self, i, o, k):
        super().__init__(); self.pad = k - 1
        self.c = nn.Conv1d(i, o, k); self.bn = nn.BatchNorm1d(o)
    def forward(self, x):
        return torch.relu(self.bn(self.c(nn.functional.pad(x, (self.pad, 0)))))   # 왼쪽만 패딩


class CNN1D(nn.Module):
    def __init__(self, nin=55, ncls=3):
        super().__init__()
        self.net = nn.Sequential(CConv(nin, 64, 5), CConv(64, 64, 5), CConv(64, 128, 3))
        self.out = nn.Linear(128, ncls)
    def forward(self, x):
        return self.out(self.net(x.transpose(1, 2)).transpose(1, 2))


class TCNTEWide(TCNTE):
    """폭 넓힌 TCNTE: TCN 채널 32/64/64, Transformer d_model 64, 헤드 4, FF 128. 나머지 구조·미래 차단은 TCNTE와 같음"""
    def __init__(self, nin=55, ncls=3):
        super().__init__(nin, ncls, ch=(32, 64, 64), d=64, heads=4, ff=128)


MODELS = {"tcnte": TCNTE, "gru": GRUNet, "cnn": CNN1D, "tcnte_wide": TCNTEWide}


class TCNTERef(nn.Module):
    """레퍼런스 논문 방식: 마스크 없는 Transformer + 시간축 평균 -> 창 하나에 2클래스 (넘어짐 / 나머지)"""
    def __init__(self, nin=55, ncls=2):
        super().__init__()
        self.tcn = nn.Sequential(TBlock(nin, 8, 5, 1), TBlock(8, 16, 5, 2), TBlock(16, 32, 5, 4))
        self.pos = nn.Parameter(torch.zeros(1, W, 32))
        nn.init.normal_(self.pos, std=0.02)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 32, 0.1, batch_first=True), 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, ncls)
    def forward(self, x):                                  # (B,T,55) -> (B,2)
        T = x.shape[1]
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2) + self.pos[:, :T]
        return self.head(self.te(h).mean(1))


MODELS["tcnte_ref"] = TCNTERef


def focal_loss(logits, y, alpha, gamma=3.0, w=None):
    """logits (N,C) softmax 전 값, y (N,) -1 = 제외, alpha (C,)
    w (N,) 프레임별 손실 배수 (B-5 --hn-weight, 기본 None = 원래와 같음)"""
    m = y >= 0
    if m.sum() == 0:
        return logits.sum() * 0
    lp = torch.log_softmax(logits[m], -1)
    yy = y[m]
    lpy = lp.gather(1, yy[:, None]).squeeze(1)
    if w is None:
        return (-alpha[yy] * (1 - lpy.exp()) ** gamma * lpy).mean()
    return (-alpha[yy] * (1 - lpy.exp()) ** gamma * lpy * w[m]).mean()


def causal_test(model, nin=55):
    """미래 프레임을 바꿔도 과거 출력이 그대로인지 + 앞부분만 넣은 결과가 같은지"""
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, W, nin)
        y1 = model(x)
        x2 = x.clone(); x2[:, 20:] = torch.randn(4, W - 20, nin) * 5
        d1 = (y1[:, :20] - model(x2)[:, :20]).abs().max().item()
        d2 = (model(x[:, :12]) - y1[:, :12]).abs().max().item()
    model.train()
    return d1, d2
