"""학습된 분류기를 영상이나 웹캠에 돌려 보기

  python demo.py 영상.mp4                         # -> out/demo/영상.mp4 (사람마다 판정을 그린 영상)
  python demo.py 영상.mp4 --show                  # 처리하면서 창으로도 보기 (q로 종료)
  python demo.py 0 --show                         # 웹캠 0번
  python demo.py 영상.mp4 --ckpt out/runs/e2_gru/best.pt

학습 때와 같게 YOLO26n-pose + ByteTrack(bytetrack.yaml), 15fps로 관절을 뽑고
infer.StreamClassifier로 사람(추적 ID)마다 매 프레임 [정상, 넘어지는 중, 쓰러짐] 확률을 낸다.
화면에 안 잡힌 사람은 3초까지 None으로 시간을 맞추고, 그 뒤에는 지운다.
상태 머신(몇 초 이상 쓰러짐이면 경보 등)은 구현팀 몫이라 여기서는 분류기 출력만 보여 준다.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from extract_pose import FPS, read_resampled
from infer import StreamClassifier

NAMES = ["Normal", "Falling", "Fallen"]
COLORS = [(80, 200, 80), (0, 165, 255), (0, 0, 255)]             # BGR: 초록, 주황, 빨강
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
KEEP_MISSING = 3 * FPS                                             # 안 보여도 상태를 유지하는 프레임 수


def frames_15fps(src):
    """(이미지)를 15fps로 내보낸다. 파일은 extract_pose와 같은 방식, 웹캠은 시각 기준으로 건너뛴다."""
    if not src.isdigit():
        for _, img, _ in read_resampled(src):
            yield img
        return
    # 카메라 읽기는 따로 돌려 항상 최신 프레임만 들고 있고, 여기서는 1/15초마다 그 프레임을 가져간다.
    # 읽기와 처리를 한 줄로 돌리면 둘의 시간이 더해져 15fps보다 느려지고, 그러면 모델이 보는 동작이 실제보다 빨라진다.
    import threading
    cap = cv2.VideoCapture(int(src))
    if not cap.isOpened():
        raise SystemExit(f"웹캠 {src}을 열 수 없음")
    latest, stop = [None], threading.Event()

    def grab():
        while not stop.is_set():
            ok, img = cap.read()
            if not ok:
                break
            latest[0] = img
    th = threading.Thread(target=grab, daemon=True)
    th.start()
    nxt = time.time()
    try:
        while th.is_alive() or latest[0] is not None:
            wait = nxt - time.time()
            if wait > 0:
                time.sleep(wait)
            nxt = max(nxt + 1 / FPS, time.time() - 1 / FPS)            # 밀리면 따라잡으려 몰아서 처리하지 않는다
            img, latest[0] = latest[0], None
            if img is None:
                if not th.is_alive():
                    break
                continue
            yield img
    finally:
        stop.set()
        th.join(timeout=1)
        cap.release()


def draw(img, kpts, box, pid, p):
    c = int(p.argmax())
    col = COLORS[c]
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    for a, b in SKELETON:
        if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
            cv2.line(img, tuple(kpts[a, :2].astype(int)), tuple(kpts[b, :2].astype(int)), col, 2)
    label = f"#{pid} {NAMES[c]} {p[c]:.2f}"
    cv2.putText(img, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for k in range(3):                                             # 박스 오른쪽에 확률 막대
        h = int(p[k] * 60)
        cv2.rectangle(img, (x2 + 4 + 10 * k, y1 + 60 - h), (x2 + 12 + 10 * k, y1 + 60), COLORS[k], -1)


def main():
    ap = argparse.ArgumentParser(description="분류기 데모: 영상/웹캠 -> 사람마다 판정을 그린 영상")
    ap.add_argument("src", help="영상 파일 경로 또는 웹캠 번호(0)")
    ap.add_argument("--ckpt", default="out/runs/e2_tcnte/best.pt")
    ap.add_argument("--out", help="출력 mp4 (기본 out/demo/<이름>.mp4)")
    ap.add_argument("--show", action="store_true", help="창으로 보기 (q로 종료)")
    ap.add_argument("--device", default="cpu", help="포즈 모델 장치: cpu | mps | cuda:0")
    ap.add_argument("--model", default="yolo26n-pose.pt")
    a = ap.parse_args()

    from ultralytics import YOLO
    pose = YOLO(a.model)
    clf = StreamClassifier(a.ckpt)
    print(f"분류기 {a.ckpt} ({clf.info['model']}, 검증 macro-F1 {clf.info.get('val_macro_f1', float('nan')):.3f})")
    out = Path(a.out or f"out/demo/{'webcam' if a.src.isdigit() else Path(a.src).stem}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer, missing, events = None, {}, {}
    t0, f = time.time(), 0
    for f, img in enumerate(frames_15fps(a.src)):
        r = pose.track(img, persist=(f > 0), tracker="bytetrack.yaml", verbose=False, device=a.device)[0]
        seen = set()
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None and r.keypoints.conf is not None:
            xy, kc = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
            bx = r.boxes.xyxy.cpu().numpy()
            for j, pid in enumerate(r.boxes.id.int().cpu().tolist()):
                kpts = np.concatenate([xy[j], kc[j][:, None]], 1)
                p = clf.step(pid, kpts, bx[j])
                draw(img, kpts, bx[j], pid, p)
                seen.add(pid)
                missing[pid] = 0
                c = int(p.argmax())
                if c > 0:
                    events.setdefault(pid, []).append((f, c))
        for pid in list(missing):                                  # 이번 프레임에 안 잡힌 사람
            if pid in seen:
                continue
            missing[pid] += 1
            if missing[pid] > KEEP_MISSING:
                clf.drop(pid)
                del missing[pid]
            else:
                clf.step(pid, None, None)
        cv2.putText(img, f"{f / FPS:6.1f}s", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        writer.write(img)
        if a.show:
            cv2.imshow("fall demo (q: quit)", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer is not None:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()

    n = f + 1
    rate = n / max(time.time() - t0, 1e-6)
    print(f"{n}프레임 ({n / FPS:.1f}초) 처리, {rate:.1f} 프레임/초 -> {out}")
    if a.src.isdigit() and rate < FPS * 0.9:
        print(f"주의: 처리가 {FPS}fps보다 느려 모델에는 동작이 실제보다 {FPS / rate:.1f}배 빠르게 보였다. "
              f"--show를 빼거나 카메라 해상도를 낮춰 보자.")
    if not events:
        print("넘어지는 중·쓰러짐으로 판정된 프레임 없음")
    for pid, ev in sorted(events.items()):                         # 사람별로 이어진 구간을 묶어 출력
        segs, (s, c0), prev = [], ev[0], ev[0][0]
        for fr, c in ev[1:] + [(None, None)]:
            if fr is None or c != c0 or fr != prev + 1:
                segs.append(f"{NAMES[c0]} {s / FPS:.1f}~{(prev + 1) / FPS:.1f}s")
                if fr is not None:
                    s, c0 = fr, c
            if fr is not None:
                prev = fr
        print(f"  #{pid}: " + ", ".join(segs))


if __name__ == "__main__":
    main()
