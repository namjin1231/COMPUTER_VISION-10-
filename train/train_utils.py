"""학습·평가 공통: 조각 꺼내기와 증강, 실시간 방식 예측, 판정 로직."""
import numpy as np
import torch
from features import flip_features

VEL = [52, 54]


def get_window(X, Y, s, rng, aug, win):
    """X (T,55), Y (T,k) 정답들 -> 시작 s의 win 프레임 조각. aug=True면 증강."""
    if aug and rng.random() < 0.3:                        # 속도 0.8~1.2배
        L = int(round(win * rng.uniform(0.8, 1.2)))
        s0 = min(s, len(X) - L) if L <= len(X) else 0
        L = min(L, len(X))
        src = np.linspace(s0, s0 + L - 1, win)
        Xw = np.stack([np.interp(src, np.arange(len(X)), X[:, k]) for k in range(X.shape[1])], 1).astype(np.float32)
        Xw[:, VEL] *= (L - 1) / (win - 1)                 # 빨리 감으면 프레임당 움직임도 커짐
        Yw = Y[np.round(src).astype(int)]
    else:
        Xw, Yw = X[s:s + win].astype(np.float32).copy(), Y[s:s + win].copy()
    if aug:
        if rng.random() < 0.5:
            Xw = flip_features(Xw)
        J = Xw[:, :51].reshape(win, 17, 3)
        if rng.random() < 0.5:
            J[:, :, :2] += rng.normal(0, 0.01, J[:, :, :2].shape) * (J[:, :, 2:3] >= 0.3)
        if rng.random() < 0.3:                            # 관절 지우기: 실전의 '직전 값 채우기'와 같은 모양
            for j in rng.choice(17, rng.integers(1, 5), replace=False):
                J[:, j, 2] = 0
                J[:, j, :2] = J[0, j, :2]
        Xw[:, :51] = J.reshape(win, 51)
    return Xw, Yw


def ref_frame_labels(y, y16):
    """레퍼런스 방식 프레임 정답: 넘어짐 1, 쓰러진 뒤 제외(-1), 사람 안 잡힘 제외(-1), 나머지(누움 포함) 0"""
    fr = (y16 == 1).astype(np.int64)
    fr[y16 == 2] = -1
    fr[(y == -1) & (y16 != 6)] = -1
    return fr


def ref_window_label(fr):
    """창 다수결: 제외 프레임이 절반 넘으면 -1, 넘어짐 프레임이 절반 넘으면 1, 아니면 0"""
    n = len(fr)
    if (fr == -1).sum() > n // 2:
        return -1
    return int((fr == 1).sum() > n // 2)


@torch.no_grad()
def stream_predict(model, X, dev, win=30):
    """프레임마다 '그 프레임까지의 최근 30프레임'만 본 확률 (T,3).
    처음 30프레임은 첫 창을 한 번 넣은 출력 (미래 차단 구조라 앞부분만 넣은 것과 같다)."""
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    T = len(X)
    first = torch.softmax(model(X[None, :min(T, win)]), -1)[0]
    if T <= win:
        return first.cpu().numpy()
    wins = X.unfold(0, win, 1).permute(0, 2, 1)[1:]        # 끝 프레임 = win..T-1
    outs = [torch.softmax(model(wins[i:i + 2048]), -1)[:, -1] for i in range(0, len(wins), 2048)]
    return torch.cat([first] + outs).cpu().numpy()


@torch.no_grad()
def stream_predict_ref(model, X, dev, win=15):
    """레퍼런스 방식: 끝 프레임 t의 창 X[t-win+1:t+1] -> P(넘어짐). 창이 안 차는 앞부분은 nan"""
    model.eval()
    T = len(X)
    p = np.full(T, np.nan, np.float32)
    if T < win:
        return p
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    wins = X.unfold(0, win, 1).permute(0, 2, 1)
    outs = [torch.softmax(model(wins[i:i + 4096]), -1)[:, 1] for i in range(0, len(wins), 4096)]
    p[win - 1:] = torch.cat(outs).cpu().numpy()
    return p


def runs_of(cond, n):
    """cond (T,) bool 이 n프레임 연속인 첫 끝 프레임 인덱스들 (bool 배열)"""
    c = np.zeros(len(cond), bool); run = 0
    for t, v in enumerate(cond):
        run = run + 1 if v else 0
        c[t] = run >= n
    return c


def speed_max(X, look=15):
    """프레임마다 최근 look프레임(+현재) 안의 특징 54번(박스 중심 아래쪽 속도) 최대값. 미래 프레임 안 씀"""
    v = X[:, 54]
    return np.array([v[max(0, t - look):t + 1].max() for t in range(len(v))], np.float32)


def confirm_frame(prob, fps=15, th_fall=0.5, n_fall=2, th_down=0.6, n_down=8, window_s=3.0, vmax=None, v_gate=None):
    """판정 로직: 의심(P넘어짐>=th 2프레임 연속) 후 3초 안에 P쓰러짐>=th 8프레임 연속 -> 쓰러짐 확인 프레임
    v_gate(B-3, 기본 None=끔): 의심으로 갈 때 vmax[t](최근 1초 특징 54 최대값) >= v_gate 도 요구"""
    sus_t = None; run_f = run_d = 0
    for t in range(len(prob)):
        run_f = run_f + 1 if prob[t, 1] >= th_fall else 0
        fast = v_gate is None or vmax[t] >= v_gate
        if run_f >= n_fall and fast and sus_t is None:
            sus_t = t
        if sus_t is not None:
            if t - sus_t > window_s * fps:
                sus_t, run_d = None, 0
                continue
            run_d = run_d + 1 if prob[t, 2] >= th_down else 0
            if run_d >= n_down:
                return t
    return None
