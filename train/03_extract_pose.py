"""3단계: 관절 추출 (YOLO26n-pose + ByteTrack). 영상마다 '모든 트랙'을 저장한다.

사용법:
  python 03_extract_pose.py --dataset GMDCSA24
  python 03_extract_pose.py --dataset of-syn              # 밤새
  python 03_extract_pose.py --dataset of-syn --shard 0/2  # GPU 메모리가 넉넉하면 창 2개로 나눠 돌리기 (0/2, 1/2)
끊겨도 같은 명령을 다시 실행하면 이미 끝난 영상은 건너뛴다.

결과: pose/<dataset>/<vid>.npz
  t (T,) 프레임 시각(초, 영상 시작=0), src_fps, track_ids (K,),
  kpts (K,T,17,2) 픽셀, conf (K,T,17), box (K,T,4) xyxy   (그 프레임에 없는 트랙은 0)
"""
import argparse, csv, os, sys, time
import numpy as np
import pandas as pd
from common import ROOT, LABELS, POSE, LOGS, FPS

MERGE_GAP = 15        # 1초 안에 다시 나타나면
MERGE_DIST = 0.5      # 마지막 박스 높이의 0.5배 이내면 같은 사람
MIN_TRACK = 3         # 3프레임 미만 트랙은 잡음으로 버림


def select_15fps(frame_iter):
    """(시각, 프레임) 스트림에서 15fps 목표 시각(0, 1/15, ...)에 가장 가까운 프레임만 내보낸다.
    프레임을 쌓아두지 않으므로 긴 영상도 메모리를 적게 쓴다."""
    k, prev, t0 = 0, None, None
    for t, fr in frame_iter:
        if t0 is None:
            t0 = t
        t -= t0
        while t >= k / FPS:
            tk = k / FPS
            if prev is not None and abs(prev[0] - tk) <= abs(t - tk):
                yield prev
            else:
                yield (t, fr)
            k += 1
        prev = (t, fr)
    if prev is not None and (k / FPS - prev[0]) <= 0.5 / FPS:
        yield prev


def frames_pyav(path):
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.average_rate) if st.average_rate else 0.0
        def gen():
            for fr in c.decode(st):
                if fr.time is None:
                    continue
                yield float(fr.time), fr
        for t, fr in select_15fps(gen()):
            yield t, fr.to_ndarray(format="bgr24"), fps


def frames_cv2(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    def gen():
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, img
    try:
        for t, img in select_15fps(gen()):
            yield t, img, fps
    finally:
        cap.release()


def run_tracker(model, frames, dev, imgsz=640):
    """frames: (t, bgr, fps) 반복자 -> t 배열, fps, {id: {i: (kpts, conf, box)}}"""
    ts, tracks, fps = [], {}, 0.0
    extra = {} if imgsz == 640 else {"imgsz": imgsz}      # 기본(640)은 원래 호출과 완전히 같게
    for i, (t, img, fps) in enumerate(frames):
        r = model.track(img, persist=(i > 0), tracker="bytetrack.yaml", verbose=False, device=dev, **extra)[0]
        ts.append(t)
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue
        ids = r.boxes.id.int().cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        for j, tid in enumerate(ids):
            tracks.setdefault(int(tid), {})[i] = (kxy[j], kc[j], xyxy[j])
    return np.array(ts, np.float32), fps, tracks


def merge_tracks(tracks):
    """끊긴 트랙 이어붙이기. 반환: 합쳐진 트랙 dict, 합친 횟수"""
    tr = {k: dict(v) for k, v in tracks.items() if v}
    merged = 0
    changed = True
    while changed:
        changed = False
        info = sorted(((min(v), max(v), k) for k, v in tr.items()))
        for sa, ea, a in info:
            best = None
            for sb, eb, b in info:
                if b == a or sb <= ea or sb - ea > MERGE_GAP:
                    continue
                la, fb = tr[a][ea][2], tr[b][sb][2]
                ca = np.array([(la[0] + la[2]) / 2, (la[1] + la[3]) / 2])
                cb = np.array([(fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2])
                h = max(la[3] - la[1], 1.0)
                if np.linalg.norm(ca - cb) < MERGE_DIST * h and (best is None or sb < best[0]):
                    best = (sb, b)
            if best:
                tr[a].update(tr.pop(best[1]))
                merged += 1
                changed = True
                break
    return tr, merged


def to_arrays(tracks, T):
    keep = {k: v for k, v in tracks.items() if len(v) >= MIN_TRACK} or tracks
    ids = sorted(keep, key=lambda k: -len(keep[k]))
    K = len(ids)
    kp = np.zeros((K, T, 17, 2), np.float32); cf = np.zeros((K, T, 17), np.float32); bx = np.zeros((K, T, 4), np.float32)
    for a, k in enumerate(ids):
        for i, (p, c, b) in keep[k].items():
            kp[a, i], cf[a, i], bx[a, i] = p, c, b
    return np.array(ids, np.int32), kp, cf, bx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default=None, help="train/val/test 중 하나만")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--weights", default="yolo26n-pose.pt", help="포즈 모델 (기본 yolo26n-pose.pt)")
    ap.add_argument("--imgsz", type=int, default=640, help="입력 크기 (기본 640)")
    ap.add_argument("--videos-csv", default="videos.csv", help="labels 아래 영상 표 (기본 videos.csv, E6: videos_itw.csv)")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    dev = 0 if torch.cuda.is_available() else "cpu"
    wpath = ROOT / a.weights
    model = YOLO(str(wpath) if wpath.exists() else a.weights)     # 폴더에 없으면 ultralytics가 자동으로 받음

    v = pd.read_csv(LABELS / a.videos_csv)
    v = v[(v.dataset == a.dataset)]
    if a.split:
        v = v[v.split == a.split]
    v["exists"] = [(ROOT / f).exists() for f in v.video_file]
    if (~v.exists).any():
        print(f"!! 파일 없는 영상 {int((~v.exists).sum())}개는 건너뜀 (다운로드 확인)")
    v = v[v.exists]
    k, n = map(int, a.shard.split("/"))
    v = v.iloc[k::n]
    out_dir = POSE / a.dataset; out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in v.itertuples() if not (out_dir / f"{r.vid}.npz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.dataset} shard {a.shard}: 대상 {len(v)}개, 남은 {len(todo)}개, device={dev}")

    log_path = LOGS / f"extract_{a.dataset}{'' if n == 1 else f'.shard{k}'}.csv"
    new = not log_path.exists()
    fh = open(log_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, ["vid", "split", "n_frames", "src_fps", "n_tracks", "merged", "secs", "decoder", "status"])
    if new:
        w.writeheader()
    t_start = time.time()
    for i, r in enumerate(todo):
        t0 = time.time()
        row = dict(vid=r.vid, split=r.split, n_frames=0, src_fps=0, n_tracks=0, merged=0, secs=0, decoder="", status="ok")
        try:
            try:
                ts, fps, tracks = run_tracker(model, frames_pyav(ROOT / r.video_file), dev, a.imgsz); row["decoder"] = "pyav"
                if len(ts) == 0:
                    raise RuntimeError("pyav 0 frames")
            except Exception:
                ts, fps, tracks = run_tracker(model, frames_cv2(ROOT / r.video_file), dev, a.imgsz); row["decoder"] = "cv2"
                if len(ts) == 0:
                    raise RuntimeError("영상을 읽을 수 없음 (0 프레임)")
            tracks, merged = merge_tracks(tracks)
            ids, kp, cf, bx = to_arrays(tracks, len(ts))
            tmp = out_dir / f"{r.vid}.tmp.npz"
            np.savez_compressed(tmp, t=ts, src_fps=np.float32(fps), track_ids=ids, kpts=kp, conf=cf, box=bx)
            os.replace(tmp, out_dir / f"{r.vid}.npz")
            row.update(n_frames=len(ts), src_fps=round(fps, 2), n_tracks=len(ids), merged=merged)
        except Exception as e:
            row["status"] = f"error:{type(e).__name__}:{e}"[:200]
            with open(LOGS / "extract_errors.csv", "a", encoding="utf-8") as ef:
                ef.write(f"{a.dataset},{r.vid},{row['status']}\n")
        row["secs"] = round(time.time() - t0, 2)
        w.writerow(row); fh.flush()
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            el = time.time() - t_start
            print(f"  [{i + 1}/{len(todo)}] {el / (i + 1):.2f}초/영상, 남은 시간 약 {el / (i + 1) * (len(todo) - i - 1) / 60:.0f}분", flush=True)
    fh.close()
    print("끝. 오류 목록:", LOGS / "extract_errors.csv")


if __name__ == "__main__":
    main()
