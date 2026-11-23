"""낙상 감시 모니터링 데모 서버.
카메라를 각각 별도 프로세스(engine.camera_process)로 돌리고, 브라우저 화면(monitor/web)으로 보여 준다.
모든 카메라가 모델을 불러오면 카운트다운 뒤 '동시에' 시작한다 (공유 시계). 클립이 끝나면 다 같이 처음부터 다시 돈다.
  - 영상: /stream/<k>  (MJPEG, 뼈대·상태 색이 그려진 프레임)
  - 상태: /events      (SSE, 0.25초마다 카메라 상태·확률·그래프·새 이벤트·시작 시각)
  - 경보 확인: POST /api/ack/<k>     - 처음부터 다시: POST /api/restart (화면에서 R 키)
파이썬 기본 라이브러리만 씀 (추가 설치 없음).
사용: python -X utf8 monitor/server.py   (브라우저가 자동으로 열림, 끄려면 Ctrl+C)
"""
import json, multiprocessing as mp, sys, threading, time, webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MON = Path(__file__).resolve().parent
WEB = MON / "web"
sys.path.insert(0, str(MON))
PORT = 8765
FPS = 15
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8",
         ".svg": "image/svg+xml", ".png": "image/png"}


class Hub:
    """카메라 프로세스에서 온 최신 프레임·상태·이벤트를 모으고, 공유 시계(sched)로 시작·재시작을 정한다."""

    def __init__(self, cfg, sched):
        self.cfg, self.sched = cfg, sched
        n = self.n = len(cfg["cameras"])
        self.jpeg = [None] * n
        self.snap = [dict(id=c["id"], name=c["name"], state="NONE", p=[1, 0, 0], fps=0, clip="", clip_k=0, clip_n=len(c["playlist"]),
                          t=0, alarm=False, alarm_since=None, alarm_epoch=None, status="연결 중", round=0) for c in cfg["cameras"]]
        self.cond = [threading.Condition() for _ in range(n)]
        self.hist = [deque(maxlen=160) for _ in range(n)]     # 그래프·타임라인용: 4프레임마다 (P넘어짐, P쓰러짐, 상태), 약 43초
        self.last_t = [None] * n
        self.ready, self.nframes = set(), []
        self.done, self.first_done = set(), None
        self.events, self.n_ev, self.lock = deque(maxlen=300), 0, threading.Lock()
        self.started = time.strftime("%H:%M:%S")

    # ---- 공유 시계: [재시작 번호, 이번 회차 시작 시각, 회차 번호] ----
    def schedule(self, delay):
        """처음 시작·재시작: delay초 뒤 1회차를 모든 카메라가 동시에 시작"""
        with self.sched.get_lock():
            self.sched[0] += 1; self.sched[1] = time.time() + delay; self.sched[2] = 0
        self.done, self.first_done = set(), None
        with self.lock:
            self.events.clear()
        for h in self.hist:
            h.clear()
        self.last_t = [None] * self.n

    def sched_info(self):
        with self.sched.get_lock():
            return int(self.sched[0]), float(self.sched[1]), float(self.sched[2])

    def collect(self, q):
        while True:
            kind, k, payload = q.get()
            if kind == "frame":
                jpg, snap = payload
                with self.cond[k]:
                    self.jpeg[k] = jpg; self.snap[k] = snap; self.cond[k].notify_all()
                key = (snap.get("round"), snap.get("t"))
                if snap.get("status") == "LIVE" and key != self.last_t[k]:
                    self.last_t[k] = key
                    if round(snap["t"] * FPS) % 4 == 0:
                        self.hist[k].append(None if snap["state"] == "NONE" else (snap["p"][1], snap["p"][2], snap["state"]))
            elif kind == "done":
                gen, t0, rnd = self.sched_info()
                if payload["gen"] == gen and payload["round"] == int(rnd):
                    self.done.add(k)
                    self.first_done = self.first_done or time.time()
                    if len(self.done) == self.n:                         # 모든 카메라가 이번 회차를 끝냄 -> 다음 회차를 동시에 시작
                        start = max(time.time() + 0.8, self.first_done + self.cfg["hold_end_s"])
                        with self.sched.get_lock():
                            self.sched[1] = start; self.sched[2] = rnd + 1
                        self.done, self.first_done = set(), None
            elif kind == "ready":
                self.ready.add(k); self.nframes.append(payload["n"])
                if len(self.ready) == self.n and self.sched_info()[1] <= 0:
                    self.schedule(self.cfg.get("countdown_s", 3))          # 모든 카메라 준비 -> 카운트다운 뒤 동시에 시작
            elif kind == "snap":
                self.snap[k] = payload
            elif kind == "event":
                with self.lock:
                    self.n_ev += 1; payload["n"] = self.n_ev; self.events.append(payload)

    def events_since(self, n):
        with self.lock:
            return [e for e in self.events if e["n"] > n]


def make_handler(hub, cmds):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                p = "/index.html"
            if p.startswith("/stream/"):
                return self.stream(int(p.split("/")[2]))
            if p == "/events":
                return self.sse()
            if p == "/api/config":
                c = hub.cfg
                body = dict(version=2, site=c["site"], ward=c["ward"], alarm_s=c["alarm_s"], started=hub.started,
                            cameras=[dict(id=x["id"], name=x["name"]) for x in c["cameras"]],
                            judge=dict(model="TCNTE ×3 (최종 R-b)", th_fall=0.5, th_down=0.4, n_fall=2, n_down=4, window_s=5, v_gate=0.02))
                return self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            f = (WEB / p.lstrip("/")).resolve()
            if WEB in f.parents and f.is_file():
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            p = self.path.split("?")[0]
            if p.startswith("/api/ack/"):
                k = int(p.split("/")[3])
                if 0 <= k < len(cmds):
                    cmds[k].put("ack")
                    return self._send(200, b'{"ok":true}', "application/json")
            if p == "/api/restart":
                if len(hub.ready) == hub.n:
                    hub.schedule(hub.cfg.get("countdown_s", 3))
                    return self._send(200, b'{"ok":true}', "application/json")
                return self._send(409, b'{"ok":false}', "application/json")
            self._send(404, b"not found", "text/plain")

        def stream(self, k):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "close"); self.end_headers()
            last = None
            try:
                while True:
                    with hub.cond[k]:
                        hub.cond[k].wait_for(lambda: hub.jpeg[k] is not None and hub.jpeg[k] is not last, timeout=2.0)
                        jpg = hub.jpeg[k]
                    if jpg is None or jpg is last:
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        def sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "close"); self.end_headers()
            n, g_seen = 0, None
            try:
                while True:
                    gen, t0, rnd = hub.sched_info()
                    if gen != g_seen:
                        g_seen, n = gen, 0
                    ev = hub.events_since(n)
                    if ev:
                        n = ev[-1]["n"]
                    msg = json.dumps(dict(cams=hub.snap, hist=[list(h) for h in hub.hist], events=ev, gen=gen, start_at=t0, round=int(rnd),
                                          server_now=time.time(), ready=len(hub.ready), n=hub.n), ensure_ascii=False)
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8")); self.wfile.flush()
                    time.sleep(0.25)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
    return H


def main():
    import engine
    cfg = engine.load_cfg()
    missing = [v for c in cfg["cameras"] for v in c["playlist"] if not (MON / "clips" / f"{v}.mp4").exists()]
    if missing:
        sys.exit(f"클립이 없습니다: {missing}\n먼저 실행: python -X utf8 monitor/prepare_clips.py")
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=64)
    cmds = [ctx.Queue() for _ in cfg["cameras"]]
    sched = ctx.Array("d", [0.0, 0.0, 0.0])                          # [재시작 번호, 이번 회차 시작 시각, 회차 번호]
    procs = [ctx.Process(target=engine.camera_process, args=(k, c, cfg, q, cmds[k], sched), daemon=True) for k, c in enumerate(cfg["cameras"])]
    for p in procs:
        p.start()
    hub = Hub(cfg, sched)
    threading.Thread(target=hub.collect, args=(q,), daemon=True).start()
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PORT
    ThreadingHTTPServer.allow_reuse_address = False                  # 이미 켜진 서버가 있으면 같은 포트를 같이 잡지 않고 바로 알림
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub, cmds))
    except OSError:
        for p in procs:
            p.terminate()
        sys.exit(f"포트 {port}를 이미 쓰고 있습니다. 이미 켜 둔 모니터 서버 창을 Ctrl+C로 끄거나, --port 8766 처럼 다른 포트를 쓰세요.")
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"모니터링 화면: {url}  (카메라 {len(procs)}대 준비 중, 모두 준비되면 동시에 시작. 처음부터 다시 = 화면에서 R. 끄려면 Ctrl+C)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
