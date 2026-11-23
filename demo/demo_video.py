r"""데모: 영상(또는 웹캠)에 낙상 감지 시스템을 실제로 돌려서, 뼈대·상태·경보를 그린 영상을 만든다.

학습과 똑같은 처리: 15fps 프레임 선택 -> YOLO26n-pose + ByteTrack -> 사람별 관절 기록 -> features.py
-> TCNTE 앙상블(최근 30프레임, 마지막 프레임 결과) -> 확률 부드럽게(최근 N프레임 평균) -> 판정 로직.

사용법:
  python demo_video.py --video raw\GMDCSA24\Subject_4\Fall\01.mp4
  python demo_video.py --video my_fall.mp4 --alarm-s 5
  python demo_video.py --webcam 0                       # 실시간 화면 (q로 종료)
결과: logs\demo\<영상이름>_demo.mp4, logs\demo\<영상이름>_frames.csv (프레임별 확률·상태)

판정 로직 (사람마다):
  정상 --P(넘어짐)>=th_fall n_fall프레임 연속--> 의심 --window_s 안에 P(쓰러짐)>=th_down n_down프레임 연속--> 쓰러짐 확인
  쓰러짐 확인 상태가 alarm_s초 지속 --> 경보.  의심이 window_s 넘으면 정상으로.
  해제: P(정상)>=0.6 이 1초 지속, 또는 몸통 각도 < 30도가 1초 지속
"""
import argparse, csv, json, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from common import ROOT, LOGS, FPS, W
from features import compute_features
from models import MODELS

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]
COLORS = {"NORMAL": (80, 200, 80), "SUSPECT": (0, 220, 255), "DOWN": (0, 140, 255), "ALARM": (40, 40, 230)}
HIST = 150          # 사람마다 최근 10초 관절을 쌓아서 특징 계산 (학습 때 영상 전체로 계산한 것과 맞추기 위해)
LOST_DROP = 45      # 3초 동안 안 보이면 그 사람 기록 삭제
MERGE_GAP, MERGE_DIST = 15, 0.5   # --stitch: 03 merge_tracks와 같은 값 (1초 안, 마지막 박스 높이의 0.5배 이내)


class Person:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames = {}                      # 프레임 번호 -> (kpts, conf, box)
        self.probs = deque(maxlen=cfg.smooth)
        self.state, self.sus_t, self.down_t = "NORMAL", None, None
        self.run_f = self.run_d = self.run_ok = 0
        self.last_seen = -1
        self.p = np.array([1.0, 0.0, 0.0])

    def add(self, i, kp, cf, bx):
        self.frames[i] = (kp, cf, bx); self.last_seen = i
        for k in [k for k in self.frames if k <= i - HIST]:
            del self.frames[k]

    def arrays(self, i):
        """첫 기록 ~ 현재 프레임까지. 안 보였던 프레임은 박스 0 (학습 때 '사람 없음'과 같은 처리)"""
        s = min(self.frames); T = i - s + 1
        kp = np.zeros((T, 17, 2), np.float32); cf = np.zeros((T, 17), np.float32); bx = np.zeros((T, 4), np.float32)
        for k, (a, b, c) in self.frames.items():
            kp[k - s], cf[k - s], bx[k - s] = a, b, c
        return kp, cf, bx

    def step(self, i, p, torso, vmax=None):
        """p: 이번 프레임 확률 (3,), torso: 몸통 각도/90"""
        c = self.cfg
        self.probs.append(p); self.p = q = np.mean(self.probs, 0)
        self.run_f = self.run_f + 1 if q[1] >= c.th_fall else 0
        upright = q[0] >= 0.6 or torso < 30 / 90
        self.run_ok = self.run_ok + 1 if upright else 0
        if self.state == "NORMAL":
            fast = getattr(c, "v_gate", None) is None or (vmax is not None and vmax >= c.v_gate)   # v_gate: 최근 16프레임 특징 54 최대값
            if self.run_f >= c.n_fall and fast:
                self.state, self.sus_t, self.run_d = "SUSPECT", i, 0
        elif self.state == "SUSPECT":
            self.run_d = self.run_d + 1 if q[2] >= c.th_down else 0
            if self.run_d >= c.n_down:
                self.state, self.down_t = "DOWN", i
                if getattr(c, "fix_release", False):
                    self.run_ok = 0                        # 해제용 '정상 자세' 시간은 쓰러짐 확인 순간부터 센다
            elif i - self.sus_t > c.window_s * FPS:
                self.state = "NORMAL"
        elif self.state in ("DOWN", "ALARM"):
            if self.state == "ALARM" and getattr(c, "latch_alarm", False):
                pass                                       # --latch-alarm: 한 번 울린 경보는 끝까지 유지 (직원 확인 전까지, 첫 경보 시점은 같음)
            elif self.run_ok >= FPS:                       # 1초 동안 정상 자세 -> 해제
                self.state = "NORMAL"
            elif self.state == "DOWN" and i - self.down_t >= c.alarm_s * FPS:
                self.state = "ALARM"
        return self.state


def load_models(paths, dev):
    ms = []
    for p in paths:
        p = Path(p)
        cfg = json.load(open(p.parent / "config.json", encoding="utf-8")) if (p.parent / "config.json").exists() else {"model": "tcnte"}
        m = MODELS[cfg["model"]]().to(dev).eval()
        m.load_state_dict(torch.load(p, map_location=dev))
        ms.append(m)
    return ms


@torch.no_grad()
def predict(models, X, dev):
    x = torch.as_tensor(X[-W:], dtype=torch.float32, device=dev)[None]
    return np.mean([torch.softmax(m(x), -1)[0, -1].cpu().numpy() for m in models], 0)


def draw(img, people, ids, cv2, frame_i, fps_txt, s=1.0):
    """s: 그리기용 축소 비율 (--out-width). 1이면 원래와 같음. 관절·박스 좌표는 원본 기준이라 s를 곱해서 그린다"""
    alarm = [tid for tid in ids if people[tid].state == "ALARM"]
    for tid in ids:
        pr = people[tid]
        kp, cf, bx = pr.frames[frame_i]
        if s != 1.0:
            kp, bx = kp * s, bx * s
        col = COLORS[pr.state]
        for a, b in EDGES:
            if cf[a] >= 0.3 and cf[b] >= 0.3:
                cv2.line(img, tuple(map(int, kp[a])), tuple(map(int, kp[b])), col, 3, cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, bx)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"ID {tid} {pr.state}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        for j, (nm, c2) in enumerate([("normal", (80, 200, 80)), ("fall", (0, 220, 255)), ("fallen", (40, 40, 230))]):
            yy = y2 + 16 + j * 16
            cv2.rectangle(img, (x1, yy - 10), (x1 + int(100 * pr.p[j]), yy), c2, -1)
            cv2.putText(img, f"{nm} {pr.p[j]:.2f}", (x1 + 105, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    if alarm:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 230), 12)
        cv2.rectangle(img, (0, 0), (w, 44), (40, 40, 230), -1)
        cv2.putText(img, f"FALL ALARM  ID {', '.join(map(str, alarm))}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"t={frame_i / FPS:5.1f}s  {fps_txt}", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw_panel(cv2, w, frame_i, hist, gt_segs, c, span_s=10.0, h=230):
    """영상 아래 그래프 패널 (그리기만). 최근 span_s초: P(넘어짐) 노랑 선, P(쓰러짐) 빨강 선, th_fall/th_down 점선,
    아래에 상태 띠(정상 초록/의심 노랑/쓰러짐 확인 주황/경보 빨강/사람 없음 회색)와 정답 띠(넘어짐 노랑/쓰러짐 빨강)"""
    P = np.full((h, w, 3), 24, np.uint8)
    L, R, T, B = 70, w - 16, 14, 150                  # 그래프 영역
    n = int(span_s * FPS); t_now = frame_i / FPS; t0 = t_now - span_s
    X = lambda sec: int(L + (R - L) * (sec - t0) / span_s)
    Y = lambda p: int(B - (B - T) * p)
    cv2.rectangle(P, (L, T), (R, B), (60, 60, 60), 1)
    for p in (0.0, 0.5, 1.0):
        cv2.putText(P, f"{p:.1f}", (L - 40, Y(p) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    for th, col in ((c.th_fall, (0, 220, 255)), (c.th_down, (60, 60, 255))):
        for x in range(L, R, 14):
            cv2.line(P, (x, Y(th)), (min(x + 7, R), Y(th)), col, 1, cv2.LINE_AA)
    k0 = max(0, len(hist) - n - 1)
    for k in range(k0 + 1, len(hist)):
        for idx, col in ((0, (0, 220, 255)), (1, (60, 60, 255))):
            a, b = hist[k - 1][idx], hist[k][idx]
            if not (np.isnan(a) or np.isnan(b)):
                cv2.line(P, (X((k - 1) / FPS), Y(a)), (X(k / FPS), Y(b)), col, 2, cv2.LINE_AA)
    sy0, sy1, gy0, gy1 = B + 14, B + 34, B + 46, B + 66
    for k in range(k0, len(hist)):
        col = COLORS.get(hist[k][2], (90, 90, 90))
        cv2.rectangle(P, (max(X(k / FPS), L), sy0), (max(X((k + 1) / FPS), L + 1), sy1), col, -1)
    for s, e, lab in gt_segs:
        if e < t0 or s > t_now:
            continue
        col = (0, 220, 255) if lab == 1 else (60, 60, 255)
        cv2.rectangle(P, (max(X(s), L), gy0), (min(X(min(e, t_now)), R), gy1), col, -1)
    cv2.putText(P, "state", (8, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(P, "GT", (8, gy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(P, f"P(fall) / th_fall {c.th_fall}", (L + 8, T + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(P, f"P(fallen) / th_down {c.th_down}", (L + 260, T + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1, cv2.LINE_AA)
    cv2.putText(P, f"last {span_s:.0f}s   now {t_now:.1f}s", (R - 190, T + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return P


SEVERITY = ["NORMAL", "SUSPECT", "DOWN", "ALARM"]
STATE_TXT = {"NONE": "no person", "NORMAL": "NORMAL", "SUSPECT": "SUSPECT", "DOWN": "FALL CONFIRMED", "ALARM": "ALARM"}


def draw_extras(img, cv2, frame_i, state_hist, dur_s, gt, label):
    """그리기만 하는 추가 표시 (판정에는 영향 없음).
    gt: (시작초, 끝초) 정답 넘어짐 구간 -> 아래쪽 타임라인 띠 (회색 바탕, 정답 구간 빨간 테두리, 지나온 상태 색, 현재 위치 흰 선)
    label: 왼쪽 위 칸 이름 (예: C1) + 지금 가장 심한 상태"""
    h, w = img.shape[:2]
    if gt is not None and dur_s > 0:
        x0, x1, y0, y1 = 12, w - 12, h - 52, h - 34
        X = lambda sec: int(x0 + (x1 - x0) * min(max(sec / dur_s, 0), 1))
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), -1)
        for k, st in enumerate(state_hist):
            if st in COLORS:
                cv2.rectangle(img, (X(k / FPS), y0 + 4), (max(X((k + 1) / FPS), X(k / FPS) + 1), y1 - 4), COLORS[st], -1)
        if gt[1] > gt[0]:
            cv2.rectangle(img, (X(gt[0]), y0 - 3), (X(gt[1]), y1 + 3), (60, 60, 255), 2)
            cv2.putText(img, "GT fall", (X(gt[0]), y0 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2, cv2.LINE_AA)
        cx = X(frame_i / FPS)
        cv2.line(img, (cx, y0 - 4), (cx, y1 + 4), (255, 255, 255), 2)
    if label is not None:
        st = state_hist[-1] if state_hist else "NONE"
        col = COLORS.get(st, (200, 200, 200))
        txt = f"{label}  {STATE_TXT[st]}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        y = 56 if st == "ALARM" else 8
        cv2.rectangle(img, (6, y), (18 + tw, y + th + 16), (0, 0, 0), -1)
        cv2.putText(img, txt, (12, y + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video"); g.add_argument("--webcam", type=int)
    ap.add_argument("--models", nargs="+", default=[f"runs/Rb_r2_sel/tcnte_seed{s}/best.pt" for s in (0, 1, 2)])   # 11-17 2차: 최종 = R-b (1차 B = X_hn --preset B, v1 = runs/E2 --preset v1)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--th-fall", type=float, default=0.5)
    ap.add_argument("--th-down", type=float, default=0.4)
    ap.add_argument("--n-fall", type=int, default=2)
    ap.add_argument("--n-down", type=int, default=4)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--v-gate", type=float, default=0.02,
                    help="속도 조건: 의심으로 갈 때 최근 16프레임(t-15~t) 특징 54번(박스 중심 아래쪽 속도) 최대값 >= v_gate. 기본 0.02 (11-17 2차 최종 R-b). 켜면 frames.csv에 vmax 열")
    ap.add_argument("--no-v-gate", dest="v_gate", action="store_const", const=None, help="속도 조건 끄기")
    ap.add_argument("--preset", choices=["final", "v1", "B"], default="final",
                    help="v1 = 처음 설정 (runs/E2 3개, th_fall 0.5, window 3초, v_gate 끔). B = 1차 최종 (runs/X_hn 3개, th_fall 0.4, window 4초, v_gate 0.02). th_down 0.4·n_down 4·smooth 3은 모두 같음")
    ap.add_argument("--alarm-s", type=float, default=3.0, help="쓰러짐 확인 후 경보까지 초 (짧은 시험 영상은 2~3, 실제 데모는 5)")
    ap.add_argument("--weights", default="yolo26n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    # 아래는 추가 옵션. --stitch(기본 켬, 사용자 결정 11-17), --fix-release·--alias-iou(데모를 오프라인과 같게 만드는 처리 수정, 기본 켬)
    # 원래 동작 = --no-stitch --no-fix-release (그러면 --alias-iou는 쓰이지 않음)
    ap.add_argument("--stitch", dest="stitch", action="store_true", default=True,
                    help="ID 이어붙이기 (기본 켬): 새 ID가 1초(15프레임) 안에 사라진 사람의 마지막 박스 근처(박스 높이 0.5배 이내)에 나타나면 "
                         "같은 사람으로 보고 관절 기록·판정 상태를 이어받음 (03 merge_tracks와 같은 규칙)")
    ap.add_argument("--no-stitch", dest="stitch", action="store_false", help="ID 이어붙이기 끄기")
    ap.add_argument("--alias-iou", type=float, default=0.5,
                    help="(--stitch일 때) 새 ID 박스가 이번 프레임에 이미 잡힌 사람 박스와 IoU >= 이 값이면 그 사람의 별칭으로 등록. "
                         "둘이 같이 잡힌 프레임은 먼저 연결된 쪽만 사용. 0 = 끔")
    ap.add_argument("--single-person", action="store_true",
                    help="(--stitch일 때) 한 명만 나오는 영상용: 지금 아무도 안 보이는데 새 ID가 나오면 가장 최근에 사라진 사람(3초 안)을 이어받음. "
                         "오프라인 04 stitch(한 명 데이터셋)와 맞추기 위함. 기본 끔")
    ap.add_argument("--fix-release", dest="fix_release", action="store_true", default=True,
                    help="쓰러짐 확인(DOWN)으로 바뀌는 순간 '정상 자세' 시간을 0부터 다시 셈 (기본 켬)")
    ap.add_argument("--no-fix-release", dest="fix_release", action="store_false", help="원래 해제 규칙 (확인 전부터 쌓인 시간 사용)")
    ap.add_argument("--latch-alarm", action="store_true",
                    help="경보 유지: ALARM이 한 번 울리면 정상 자세가 돼도 끄지 않음 (실제 시설처럼 직원 확인 전까지). 기본 끔. 경보 전 단계·첫 경보 시점은 같음")
    ap.add_argument("--panel", action="store_true",
                    help="영상 아래 그래프 패널: 최근 10초 P(넘어짐)·P(쓰러짐) 선, th 점선, 상태 띠, 정답 띠 (그리기만)")
    ap.add_argument("--gt-file", default=None, help="정답 구간 csv (start,end,label: 1=넘어짐, 2=쓰러짐) -> 패널 정답 띠 (그리기만)")
    ap.add_argument("--out-width", type=int, default=0, help="그리기·저장만 이 폭으로 축소 (YOLO에는 원본 프레임). 0 = 원본 크기")
    ap.add_argument("--gt", type=float, nargs=2, default=None, metavar=("START_S", "END_S"),
                    help="정답 넘어짐 구간(초) -> 아래쪽 타임라인 띠로 그리기만 함. 비낙상은 0 0")
    ap.add_argument("--label", default=None, help="왼쪽 위에 칸 이름(예: C1)과 현재 상태 표시 (격자 영상용)")
    a = ap.parse_args()
    if a.preset == "v1":
        a.models = [f"runs/E2/tcnte_seed{s}/best.pt" for s in (0, 1, 2)]; a.th_fall, a.window_s, a.v_gate = 0.5, 3.0, None
    elif a.preset == "B":
        a.models = [f"runs/X_hn/tcnte_seed{s}/best.pt" for s in (0, 1, 2)]; a.th_fall, a.window_s, a.v_gate = 0.4, 4.0, 0.02
    a.models = [str(ROOT / m) if not Path(m).is_absolute() else m for m in a.models]

    import cv2
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(ROOT / a.weights) if (ROOT / a.weights).exists() else a.weights)
    models = load_models(a.models, dev)
    print(f"모델 {len(models)}개, device={dev}, 판정값 th_fall={a.th_fall} th_down={a.th_down} n_down={a.n_down} smooth={a.smooth}")

    if a.video:
        mod = __import__("importlib").import_module("03_extract_pose")
        frames = (img for _, img, _ in mod.frames_pyav(ROOT / a.video if not Path(a.video).is_absolute() else a.video))
        name = Path(a.video).stem if not a.out else Path(a.out).stem
    else:
        cap = cv2.VideoCapture(a.webcam)
        def cam():
            t0, k = time.time(), 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if time.time() - t0 >= k / FPS:          # 15fps로 맞춰서 분류기에 넣기
                    k += 1
                    yield img
        frames = cam(); name = f"webcam{a.webcam}"
    out_dir = LOGS / "demo"; out_dir.mkdir(parents=True, exist_ok=True)
    vw = None
    fcsv = open(out_dir / f"{name}_frames.csv", "w", newline="", encoding="utf-8")
    cw = csv.writer(fcsv); cw.writerow(["frame", "t", "id", "p_normal", "p_fall", "p_fallen", "state"] + (["vmax"] if a.v_gate is not None else []))
    dur_s = 0.0
    if a.gt is not None and a.video:
        _c = cv2.VideoCapture(str(ROOT / a.video if not Path(a.video).is_absolute() else a.video))
        dur_s = _c.get(cv2.CAP_PROP_FRAME_COUNT) / max(_c.get(cv2.CAP_PROP_FPS), 1e-6); _c.release()
    alias, state_hist = {}, []                    # alias: ByteTrack ID -> 이어받은 사람 ID (--stitch)
    n_link = {"gap": 0, "iou": 0, "single": 0}    # 이어붙이기 횟수 (끊긴 뒤 이어받기 / 겹친 ID 별칭 / 한 명 영상 이어받기)
    gt_segs = []
    if a.gt_file:
        with open(a.gt_file, encoding="utf-8") as fh:
            gt_segs = [(float(r["start"]), float(r["end"]), int(r["label"])) for r in csv.DictReader(fh)]
    hist = []                                     # 패널용: 프레임마다 (p_fall, p_fallen, 상태) — 가장 심한 상태의 사람
    people, t_last, n_alarm = {}, time.time(), set()
    for i, img in enumerate(frames):
        r = yolo.track(img, persist=(i > 0), tracker="bytetrack.yaml", verbose=False, device=0 if dev == "cuda" else "cpu")[0]
        ids = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            kxy = r.keypoints.xy.cpu().numpy(); kc = r.keypoints.conf.cpu().numpy(); xyxy = r.boxes.xyxy.cpu().numpy()
            raw_ids = [int(t) for t in r.boxes.id.int().cpu().numpy()]
            seen_now = {alias.get(t, t) for t in raw_ids}          # 이번 프레임에 이미 보이는 사람은 이어받기 후보에서 뺌
            order = list(range(len(raw_ids)))
            if a.stitch and a.alias_iou > 0:                        # 아는 ID 먼저, 별칭, 새 ID 순 (겹친 ID는 먼저 연결된 쪽만 쓰기 위해)
                order.sort(key=lambda j: 0 if (raw_ids[j] in people and raw_ids[j] not in alias) else (1 if raw_ids[j] in alias else 2))
            for j in order:
                tid = raw_ids[j]
                if a.stitch:
                    if tid in alias:
                        tid = alias[tid]
                    elif tid not in people:
                        if a.alias_iou > 0:
                            hit = next((k for k in ids if box_iou(people[k].frames[i][2], xyxy[j]) >= a.alias_iou), None)
                            if hit is not None:
                                alias[tid] = hit; n_link["iou"] += 1
                                continue                        # 이번 프레임은 먼저 연결된 쪽(원래 ID)만 사용
                        b = xyxy[j]; cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); best = None
                        for k, p in people.items():
                            if k in ids or k in seen_now or not (0 < i - p.last_seen <= MERGE_GAP):
                                continue
                            la = p.frames[p.last_seen][2]
                            ca = np.array([(la[0] + la[2]) / 2, (la[1] + la[3]) / 2])
                            if np.linalg.norm(ca - cb) < MERGE_DIST * max(la[3] - la[1], 1.0) and (best is None or p.last_seen > best[0]):
                                best = (p.last_seen, k)
                        if best:
                            alias[tid] = best[1]; tid = best[1]; n_link["gap"] += 1
                        elif a.single_person and not (seen_now & set(people)):
                            # 한 명만 나오는 영상 (04 stitch와 같은 생각): 지금 아무도 안 보이는데 새 ID가 나오면
                            # 거리·간격(최대 3초 = 기록 삭제 전)과 상관없이 가장 최근에 사라진 사람을 이어받음
                            last = max(((p.last_seen, k) for k, p in people.items() if k not in ids), default=None)
                            if last is not None:
                                alias[tid] = last[1]; tid = last[1]; n_link["single"] += 1
                    if tid in ids:                      # 같은 프레임에 같은 사람으로 두 번 붙는 일은 막음
                        continue
                people.setdefault(tid, Person(a)).add(i, kxy[j], kc[j], xyxy[j])
                ids.append(tid)
        if a.stitch and a.single_person and ids:
            # 한 명 영상: 방금 사라진 사람 k가 있고, k보다 늦게 생겨 k와 같이 잡힌 프레임이 1초 이하인 ID j가 지금 보이면
            # j를 k의 별칭으로 바꿈 (04 stitch: 고른 트랙이 빈 프레임을 겹침 적은 다른 트랙으로 채움). k의 기록·판정 상태를 이어감
            done = set()
            for k in [k for k, p in people.items() if k not in ids and 0 < i - p.last_seen <= MERGE_GAP]:
                pk = people[k]
                cand = [j for j in ids if j not in done and min(people[j].frames) > min(pk.frames)
                        and len(set(people[j].frames) & set(pk.frames)) <= MERGE_GAP]
                if not cand:
                    continue
                la = pk.frames[pk.last_seen][2]
                j = min(cand, key=lambda j: np.hypot((people[j].frames[i][2][0] + people[j].frames[i][2][2] - la[0] - la[2]) / 2,
                                                     (people[j].frames[i][2][1] + people[j].frames[i][2][3] - la[1] - la[3]) / 2))
                pk.add(i, *people[j].frames[i])
                for raw, key in list(alias.items()):
                    if key == j:
                        alias[raw] = k
                alias[j] = k; del people[j]; ids[ids.index(j)] = k; n_link["single"] += 1; done.add(k)
        for tid in ids:
            pr = people[tid]
            X = compute_features(*pr.arrays(i))
            vmax = float(X[-16:, 54].max())
            st = pr.step(i, predict(models, X, dev), X[-1, 51], vmax)
            if st == "ALARM":
                n_alarm.add(tid)
            cw.writerow([i, round(i / FPS, 3), tid, *np.round(pr.p, 4), st] + ([round(vmax, 4)] if a.v_gate is not None else []))
        for tid in [t for t, p in people.items() if i - p.last_seen > LOST_DROP]:
            del people[tid]
        if alias:
            alias = {k: v for k, v in alias.items() if v in people}
        now = time.time(); fps_txt = f"{1 / max(now - t_last, 1e-6):4.1f} fps"; t_last = now
        if a.out_width:
            s = a.out_width / img.shape[1]
            vis = cv2.resize(img, (a.out_width, int(round(img.shape[0] * s / 2)) * 2), interpolation=cv2.INTER_AREA)
            vis = draw(vis, people, ids, cv2, i, fps_txt, s)
        else:
            vis = draw(img.copy(), people, ids, cv2, i, fps_txt)
        if a.gt is not None or a.label is not None:
            state_hist.append(max((people[t].state for t in ids), key=SEVERITY.index) if ids else "NONE")
            vis = draw_extras(vis, cv2, i, state_hist, dur_s, a.gt, a.label)
        if a.panel:
            if ids:
                m = max(ids, key=lambda t: (SEVERITY.index(people[t].state), len(people[t].frames)))
                hist.append((float(people[m].p[1]), float(people[m].p[2]), people[m].state))
            else:
                hist.append((np.nan, np.nan, "NONE"))
            vis = np.vstack([vis, draw_panel(cv2, vis.shape[1], i, hist, gt_segs, a)])
        if vw is None:
            h, w = vis.shape[:2]
            vw = cv2.VideoWriter(str(out_dir / f"{name}_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        vw.write(vis)
        if a.webcam is not None:
            cv2.imshow("fall demo", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if vw is not None:
        vw.release()
    fcsv.close()
    if a.stitch:
        json.dump(dict(stitch_gap=n_link["gap"], stitch_iou=n_link["iou"], stitch_single=n_link["single"], alarm_ids=sorted(n_alarm)),
                  open(out_dir / f"{name}_info.json", "w"), indent=1)
    print(f"끝: {out_dir / (name + '_demo.mp4')}  (경보 울린 사람 ID: {sorted(n_alarm) or '없음'}"
          + (f", 이어붙이기 {n_link['gap']}회 + 겹친 ID {n_link['iou']}회 + 한 명 이어받기 {n_link['single']}회)" if a.stitch else ")"))


if __name__ == "__main__":
    main()
