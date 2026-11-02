import torch.nn as nn

class FallGRU(nn.Module):
    def __init__(self, in_dim=55, hidden=128, n_cls=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.LayerNorm(128))
        self.gru = nn.GRU(128, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_cls))

    def forward(self, x):  # x: (B, 30, 55)
        h, _ = self.gru(self.inp(x))
        return self.head(h[:, -1])