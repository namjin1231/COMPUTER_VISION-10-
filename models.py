"""④ 분류기 (제안서 4.3 ④). 세 모델 모두 (B, T, 55) -> (B, T, 3) logit.
프레임 t의 출력은 t 이전 프레임만 본다 (미래 차단). 실시간에서는 마지막 프레임 출력을 쓴다.

  tcnte  메인. TCN 3블록(필터 8·16·32, 커널 5, dilation 1·2·4) + Transformer 인코더(헤드 4, FF 32)
  gru    비교용. GRU 2층(128)
  cnn    비교용. 1D-CNN 3층(64, 커널 5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT, NCLS, WIN = 55, 3, 30


class CausalConv(nn.Module):
    """왼쪽에만 패딩해서 출력 t가 입력 t 이후를 보지 않게 한 1D 합성곱. 입력·출력 (B, C, T)"""

    def __init__(self, c_in, c_out, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv(c_in, c_out, k, dilation)
        self.c2 = CausalConv(c_out, c_out, k, dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.c1(x)))
        h = self.drop(F.relu(self.c2(h)))
        return F.relu(h + self.skip(x))


class TCNTE(nn.Module):
    def __init__(self, in_dim=N_FEAT, n_cls=NCLS, filters=(8, 16, 32), kernel=5, dilations=(1, 2, 4),
                 heads=4, ff=32, dropout=0.1, max_len=WIN):
        super().__init__()
        chans = (in_dim,) + tuple(filters)
        self.tcn = nn.Sequential(*[TCNBlock(chans[i], chans[i + 1], kernel, dilations[i], dropout)
                                   for i in range(len(filters))])
        d = filters[-1]
        self.pos = nn.Parameter(torch.zeros(max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):                                   # (B, T, 55), T <= max_len
        T = x.shape[1]
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)   # 미래 프레임 차단
        return self.head(self.enc(h, src_mask=mask))        # softmax는 손실 쪽에서 한 번만


class GRUNet(nn.Module):
    def __init__(self, in_dim=N_FEAT, n_cls=NCLS, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.head(self.gru(x)[0])


class CNN1D(nn.Module):
    def __init__(self, in_dim=N_FEAT, n_cls=NCLS, width=64, kernel=5, dropout=0.1):
        super().__init__()
        self.c = nn.ModuleList([CausalConv(in_dim, width, kernel), CausalConv(width, width, kernel),
                                CausalConv(width, width, kernel)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_cls)

    def forward(self, x):
        h = x.transpose(1, 2)
        for c in self.c:
            h = self.drop(F.relu(c(h)))
        return self.head(h.transpose(1, 2))


MODELS = {"tcnte": TCNTE, "gru": GRUNet, "cnn": CNN1D}


def build_model(name):
    return MODELS[name]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def save_ckpt(path, model, name, **extra):
    torch.save(dict(model=name, state=model.state_dict(), n_feat=N_FEAT, win=WIN,
                    classes=["normal", "falling", "fallen"], **extra), path)


def load_ckpt(path):
    """best.pt -> (eval 모드 모델, 저장된 정보 dict)"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck
