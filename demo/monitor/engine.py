"""모니터링 엔진: 카메라마다 프로세스 1개가 재생 목록을 15fps로 '실시간처럼' 흘리며 실제 판정을 돌린다.
판정은 demo_video.py와 같다: YOLO26n-pose + ByteTrack -> 사람별 관절 기록 -> features 55개 -> TCNTE 3개 평균
-> smooth 3 -> 2단계 판정. Person·predict·load_models·box_iou는 demo_video.py에서 그대로 가져다 쓰고(파일은 안 고침),
프레임마다 사람 ID 잇기(--stitch --single-person)는 demo_video.main()의 루프를 그대로 옮겼다.
설정: 최종 R-b 판정값 + 경보 유지(latch) + 경보 대기 config.json alarm_s.
카메라마다 프로세스를 따로 쓰는 이유: 스레드 4개로는 파이썬 GIL 때문에 카메라당 11fps밖에 안 나왔다.
"""
import json, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import demo_video as dv                      # noqa: E402  (main은 실행되지 않음)
from features import compute_features        # noqa: E402
from common import FPS                       # noqa: E402

MON = Path(__file__).resolve().parent
SEVERITY = ["NONE", "NORMAL", "SUSPECT", "DOWN", "ALARM"]
# BGR. 웹 화면 색과 맞춤: 정상 초록, 의심 노랑, 쓰러짐 확인 주황, 경보 빨강
COL = {"NORMAL": (94, 197, 34), "SUSPECT": (21, 204, 250), "DOWN": (60, 146, 251), "ALARM": (68, 68, 239)}


def load_cfg():
    return json.load(open(MON / "config.json", encoding="utf-8"))


def judge_cfg(alarm_s):
    """최종 R-b 판정값 (logs/r2_select.json, demo_video.py 기본값과 같음) + 데모 옵션"""
    return SimpleNamespace(smooth=3, th_fall=0.5, th_down=0.4, n_fall=2, n_down=4, window_s=5.0, v_gate=0.02,
                           alarm_s=alarm_s, stitch=True, alias_iou=0.5, single_person=True, fix_release=True, latch_alarm=True)


class Tracker:
    """한 클립 동안의 사람 추적·판정 상태. demo_video.main()의 프레임 루프와 같은 처리."""

    def __init__(self, a, yolo, models, yolo_dev, model_dev):
        self.a, self.yolo, self.models, self.yolo_dev, self.model_dev = a, yolo, models, yolo_dev, model_dev
        self.people, self.alias, self.i = {}, {}, 0

    def step(self, img):
        a, i, people, alias = self.a, self.i, self.people, self.alias
        r = self.yolo.track(img, persist=(i > 0), tracker="bytetrack.yaml", verbose=False,
                            device=0 if self.yolo_dev == "cuda" else "cpu")[0]
        ids = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            kxy = r.keypoints.xy.cpu().numpy(); kc = r.keypoints.conf.cpu().numpy(); xyxy = r.boxes.xyxy.cpu().numpy()
            raw_ids = [int(t) for t in r.boxes.id.int().cpu().numpy()]
            seen_now = {alias.get(t, t) for t in raw_ids}
            order = list(range(len(raw_ids)))
            order.sort(key=lambda j: 0 if (raw_ids[j] in people and raw_ids[j] not in alias) else (1 if raw_ids[j] in alias else 2))
            for j in order:
                tid = raw_ids[j]
                if tid in alias:
                    tid = alias[tid]
                elif tid not in people:
                    hit = next((k for k in ids if dv.box_iou(people[k].frames[i][2], xyxy[j]) >= a.alias_iou), None)
                    if hit is not None:
                        alias[tid] = hit
                        continue
                    b = xyxy[j]; cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); best = None
                    for k, p in people.items():
                        if k in ids or k in seen_now or not (0 < i - p.last_seen <= dv.MERGE_GAP):
                            continue
                        la = p.frames[p.last_seen][2]
                        ca = np.array([(la[0] + la[2]) / 2, (la[1] + la[3]) / 2])
                        if np.linalg.norm(ca - cb) < dv.MERGE_DIST * max(la[3] - la[1], 1.0) and (best is None or p.last_seen > best[0]):
                            best = (p.last_seen, k)
                    if best:
                        alias[tid] = best[1]; tid = best[1]
                    elif a.single_person and not (seen_now & set(people)):
                        last = max(((p.last_seen, k) for k, p in people.items() if k not in ids), default=None)
                        if last is not None:
                            alias[tid] = last[1]; tid = last[1]
                if tid in ids:
                    continue
                people.setdefault(tid, dv.Person(a)).add(i, kxy[j], kc[j], xyxy[j])
                ids.append(tid)
        if a.single_person and ids:
            done = set()
            for k in [k for k, p in people.items() if k not in ids and 0 < i - p.last_seen <= dv.MERGE_GAP]:
                pk = people[k]
                cand = [j for j in ids if j not in done and min(people[j].frames) > min(pk.frames)
                        and len(set(people[j].frames) & set(pk.frames)) <= dv.MERGE_GAP]
                if not cand:
                    continue
                la = pk.frames[pk.last_seen][2]
                j = min(cand, key=lambda j: np.hypot((people[j].frames[i][2][0] + people[j].frames[i][2][2] - la[0] - la[2]) / 2,
                                                     (people[j].frames[i][2][1] + people[j].frames[i][2][3] - la[1] - la[3]) / 2))
                pk.add(i, *people[j].frames[i])
                for raw, key in list(alias.items()):
                    if key == j:
                        alias[raw] = k
                alias[j] = k; del people[j]; ids[ids.index(j)] = k; done.add(k)
        for tid in ids:
            pr = people[tid]
            X = compute_features(*pr.arrays(i))
            vmax = float(X[-16:, 54].max())
            pr.step(i, dv.predict(self.models, X, self.model_dev), X[-1, 51], vmax)
        for tid in [t for t, p in people.items() if i - p.last_seen > dv.LOST_DROP]:
            del people[tid]
        if alias:
            self.alias = {k: v for k, v in alias.items() if v in people}
        self.i += 1
        return ids

    def acknowledge(self):
        """직원이 경보를 확인: 경보·쓰러짐 확인 상태를 정상으로 되돌림 (그 뒤 다시 넘어지면 다시 감지)"""
        for p in self.people.values():
            if p.state in ("ALARM", "DOWN"):
                p.state, p.sus_t, p.down_t, p.run_f, p.run_d = "NORMAL", None, None, 0, 0


def draw(img, people, ids, i, cv2, s=1.0):
    """얇은 뼈대 + 관절 점 + 모서리 괄호 박스 (상태 색). 한글 글자는 웹 화면에서 보여 준다. s: 그리기 축소 비율"""
    for tid in ids:
        pr = people[tid]
        kp, cf, bx = pr.frames[i]
        kp, bx = kp * s, bx * s
        col = COL[pr.state]
        for a_, b_ in dv.EDGES:
            if cf[a_] >= 0.3 and cf[b_] >= 0.3:
                cv2.line(img, tuple(map(int, kp[a_])), tuple(map(int, kp[b_])), col, 2, cv2.LINE_AA)
        for k in range(17):
            if cf[k] >= 0.3:
                cv2.circle(img, tuple(map(int, kp[k])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, bx); L = max(12, int(min(x2 - x1, y2 - y1) * 0.18))
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * L, py), col, 3, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * L), col, 3, cv2.LINE_AA)
        tag = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(th + 12, y1)
        cv2.rectangle(img, (x1, y0 - th - 12), (x1 + tw + 12, y0 - 2), col, -1, cv2.LINE_AA)
        cv2.putText(img, tag, (x1 + 6, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
    return img


class CamRunner:
    """카메라 1대. 모든 카메라가 같은 시계(sched)를 따라 동시에 재생한다.
    sched: 공유 배열 [gen, t0, round] — gen: 재시작 번호, t0: 이번 회차 시작 시각(time.time), round: 회차 번호. t0=0이면 아직 시작 전.
    모든 카메라가 t0에 같이 시작하고, 프레임 f는 t0 + f/15초에 처리한다 (밀려도 프레임은 버리지 않음).
    회차를 끝낸 카메라는 'done'을 보내고 마지막 장면을 유지한다. 모두 끝나면 서버가 다음 회차 t0를 정한다 (느린 카메라가 있어도 같이 시작).
    out(kind, payload): 'ready' {n} | 'done' {gen, round} | 'frame' (jpeg, snapshot) | 'event' dict. cmd: 명령 큐 ('ack')."""

    def __init__(self, idx, cam, cfg, out, cmd=None, sched=None, headless=False):
        import cv2
        from ultralytics import YOLO
        torch.set_num_threads(2)
        self.cv2, self.idx, self.cam, self.cfg, self.out, self.cmd, self.sched = cv2, idx, cam, cfg, out, cmd, sched
        self.headless = headless
        self.yolo_dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(str(ROOT / "yolo26n-pose.pt"))
        # 판정 모델(TCNTE 3개)은 CPU: 모델이 작아 CPU가 더 빠름 (logs/present_metrics.json: CPU 3.9ms vs GPU 6.7ms)
        self.models = dv.load_models([str(ROOT / f"runs/Rb_r2_sel/tcnte_seed{s}/best.pt") for s in (0, 1, 2)], "cpu")
        self.a = judge_cfg(cfg["alarm_s"])
        self.clip_k, self.clip = 0, cam["playlist"][0]
        self.round = 0
        self._reset_view()
        self.status = "연결 중"
        self.log = []

    def _reset_view(self):
        self.state, self.p, self.fps, self.t = "NONE", [1.0, 0.0, 0.0], 0.0, 0.0
        self.alarm_active, self.alarm_since, self.alarm_epoch = False, None, None

    def snapshot(self):
        return dict(id=self.cam["id"], name=self.cam["name"], state=self.state, p=[round(float(x), 3) for x in self.p],
                    fps=round(self.fps, 1), clip=self.clip, clip_k=self.clip_k + 1, clip_n=len(self.cam["playlist"]),
                    t=round(self.t, 2), alarm=self.alarm_active, alarm_since=self.alarm_since, alarm_epoch=self.alarm_epoch,
                    status=self.status, round=self.round)

    def _emit(self, kind, text, **kw):
        self.out("event", dict(cam=self.idx, cam_id=self.cam["id"], name=self.cam["name"], kind=kind, text=text,
                               clip=self.clip, time=time.strftime("%H:%M:%S"), epoch=time.time(), **kw))

    def _acked(self):
        got = False
        if self.cmd is not None:
            try:
                while True:
                    if self.cmd.get_nowait() == "ack":
                        got = True
            except Exception:
                pass
        return got

    def _ack_alarm(self):
        if self.alarm_active:
            resp = round(time.time() - self.alarm_epoch, 1) if self.alarm_epoch else None
            self.alarm_active, self.alarm_since, self.alarm_epoch = False, None, None
            self._emit("ACK", "경보 확인 처리", resp_s=resp)

    def _read_sched(self):
        with self.sched.get_lock():
            return int(self.sched[0]), float(self.sched[1]), float(self.sched[2])

    def _frame0(self, vid):
        cap = self.cv2.VideoCapture(str(MON / "clips" / f"{vid}.mp4")); ok, img = cap.read(); cap.release()
        return img if ok else None

    def _publish(self, img, people=None, ids=(), i=0):
        cv2 = self.cv2
        w = self.cfg["stream_width"]; s = w / img.shape[1]
        vis = cv2.resize(img, (w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        if people is not None:
            vis = draw(vis, people, ids, i, cv2, s)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self.out("frame", (buf.tobytes(), self.snapshot()))
            return buf.tobytes()

    def run(self):
        """서버용: 준비(모델 예열) -> 'ready' 알림 -> 공유 시계를 따라 회차마다 재생"""
        pl = self.cam["playlist"]
        img0 = self._frame0(pl[0])
        Tracker(self.a, self.yolo, self.models, self.yolo_dev, "cpu").step(img0)      # 예열 (CUDA 초기화), 결과는 버림
        n = max(int(self.cv2.VideoCapture(str(MON / "clips" / f"{v}.mp4")).get(self.cv2.CAP_PROP_FRAME_COUNT)) for v in pl)
        self.status = "준비 완료"
        self.out("ready", dict(n=n))
        gen_now, done_round, last = -1, -1, None
        while True:
            gen, t0, rnd = self._read_sched()
            rnd = int(rnd)
            if gen != gen_now:                                   # 처음 시작 또는 재시작
                gen_now, done_round, last = gen, -1, None
                self._reset_view(); self.round = 0
            holding = done_round >= 0 and last is not None
            if t0 <= 0 or time.time() < t0 or rnd <= done_round:
                if holding:                                      # 회차 끝: 모든 카메라가 끝날 때까지 마지막 장면 유지
                    self.status = "다음 영상 준비"
                    if self._acked():
                        self._ack_alarm()
                    self.out("frame", (last, self.snapshot())); time.sleep(1.0 / FPS)
                else:                                            # 시작 전: 첫 장면을 멈춘 채로 보여 줌
                    self.clip_k, self.clip = 0, pl[0]
                    self.status = "준비 완료" if t0 <= 0 else "시작 대기"
                    self._publish(img0); time.sleep(0.2)
                continue
            if self.cfg.get("continuous", True):
                # 이어서 재생: 시작만 모든 카메라가 같이 하고, 그 뒤로는 카메라마다 다음 영상으로 바로 넘어감 (멈춤 없음)
                r, base = 0, t0
                while True:
                    self.round = r; self.clip_k = r % len(pl); self.clip = pl[self.clip_k]
                    if self.play(self.clip, base, gen) is None:
                        break                                    # 처음부터 다시 (R)
                    r += 1; base = time.time()
                continue
            self.round = rnd; self.clip_k = rnd % len(pl); self.clip = pl[self.clip_k]
            res = self.play(self.clip, t0, gen)
            if res is not None:                                  # 끝까지 재생함 -> 서버에 알림 (모두 끝나면 서버가 다음 회차 시작 시각을 정함)
                last = res or last; done_round = rnd
                self.out("done", dict(gen=gen, round=rnd))

    def play(self, vid, base, gen):
        """한 회차 재생. 끝까지 재생하면 마지막 화면 jpeg(없으면 b'')를, 재시작 요청으로 멈추면 None을 돌려준다."""
        cv2 = self.cv2
        cap = cv2.VideoCapture(str(MON / "clips" / f"{vid}.mp4"))
        tracker = Tracker(self.a, self.yolo, self.models, self.yolo_dev, "cpu")
        self.status = "LIVE"; prev = "NONE"; last = None; t_last = time.perf_counter(); f = 0
        if self.alarm_active and self.cfg.get("clear_alarm_on_next_clip", True):
            # 데모 전용: 다음 회차는 처음부터 다시 도는 것이라, 확인 안 한 경보를 넘기지 않음 (실제 현장은 직원 확인 전까지 유지)
            self.alarm_active, self.alarm_since, self.alarm_epoch = False, None, None
        while True:
            if self.sched is not None and self._read_sched()[0] != gen:
                cap.release(); return                            # 재시작 요청
            if not self.headless:
                d = base + f / FPS - time.time()
                if d > 0:
                    time.sleep(d)
            ok, img = cap.read()
            if not ok:
                break
            if self._acked():
                tracker.acknowledge(); self._ack_alarm()
            ids = tracker.step(img)
            i = tracker.i - 1; people = tracker.people
            st = max((people[t].state for t in ids), key=SEVERITY.index) if ids else "NONE"
            if ids:
                m = max(ids, key=lambda t: (SEVERITY.index(people[t].state), len(people[t].frames)))
                self.p = people[m].p.tolist()
            if SEVERITY.index(st) > SEVERITY.index(prev) and st in ("SUSPECT", "DOWN", "ALARM"):
                self._emit(st, {"SUSPECT": "넘어짐 의심", "DOWN": "쓰러짐 확인", "ALARM": "낙상 경보"}[st])
                if st == "ALARM" and not self.alarm_active:
                    self.alarm_active, self.alarm_since, self.alarm_epoch = True, time.strftime("%H:%M:%S"), time.time()
            elif prev in ("SUSPECT", "DOWN") and st in ("NORMAL", "NONE"):
                self._emit("CLEAR", "정상으로 돌아옴")
            prev = st; self.state, self.t = st, i / FPS
            self.log.append((vid, i, st))
            now = time.perf_counter()
            inst = 1 / max(now - t_last, 1e-6); self.fps = inst if not self.fps else 0.85 * self.fps + 0.15 * inst; t_last = now
            if not self.headless:
                last = self._publish(img, people, ids, i) or last
            f += 1
        cap.release()
        return last or b""


def camera_process(idx, cam, cfg, q, cmd, sched):
    """서버가 띄우는 카메라 프로세스. q로 (kind, idx, payload)를 보낸다. 큐가 차면 화면 프레임만 건너뜀 (판정은 계속)."""
    import queue as _q

    def out(kind, payload):
        try:
            q.put_nowait((kind, idx, payload))
        except _q.Full:
            if kind != "frame":
                q.put((kind, idx, payload))
    CamRunner(idx, cam, cfg, out, cmd, sched).run()


if __name__ == "__main__":
    # 확인용 (화면 없음): python -X utf8 monitor/engine.py --check -> 모든 클립의 첫 의심·쓰러짐 확인·경보 시각
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        cfg = load_cfg()
        for k, c in enumerate(cfg["cameras"]):
            r = CamRunner(k, c, cfg, lambda *x: None, headless=True)
            for vid in c["playlist"]:
                r.play(vid, 0, 0)
            by = {}
            for vid, i, st in r.log:
                d = by.setdefault(vid, {"SUSPECT": None, "DOWN": None, "ALARM": None})
                for s_ in ("SUSPECT", "DOWN", "ALARM"):
                    if SEVERITY.index(st) >= SEVERITY.index(s_) and d[s_] is None:
                        d[s_] = round(i / FPS, 3)
            for vid, d in by.items():
                print(c["id"], c["name"], vid, d)
