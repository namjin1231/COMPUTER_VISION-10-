"""① 관절 추출 (제안서 4.3 ①): 영상 -> YOLO26n-pose + ByteTrack -> 15fps, 영상별 모든 트랙 저장

사용
  python extract_pose.py --dataset GMDCSA24          # data/omnifall/GMDCSA24/video -> out/pose/GMDCSA24
  python extract_pose.py clip.mp4 --out out/pose/my  # 임의의 영상 또는 폴더

출력 (영상 1개당 npz, 이름은 영상 경로의 '/'를 '__'로 바꾼 것)
  n_frames, fps(15), src_fps, src_idx (n_frames,), width, height
  luma (n_frames,)   프레임 평균 밝기 (낮·밤 확인용)
  det_frame (M,), det_tid (M,), det_kpts (M,17,3), det_box (M,4), det_conf (M,)   모든 사람의 모든 검출
피험자 선택은 select_subject.py가 따로 한다. 규칙을 바꿔도 다시 추출할 필요가 없다.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

FPS = 15
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DATA = Path("data/omnifall")


def pose_name(rel_path):
    """'Subject_1/Fall/01' -> 'Subject_1__Fall__01'"""
    return "__".join(Path(rel_path).with_suffix("").parts)


def target_frames(n_src, src_fps, fps=FPS):
    """15fps 각 시각에 가장 가까운 원본 프레임 번호"""
    if n_src <= 0:
        return np.zeros(0, int)
    n = int(np.floor((n_src - 1) / src_fps * fps)) + 1
    return np.minimum(np.round(np.arange(n) * src_fps / fps).astype(int), n_src - 1)


def read_resampled(path, fps=FPS):
    """(원본 프레임 번호, 이미지)를 15fps로 내보낸다. 원본이 더 느리면 같은 프레임이 반복된다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없음: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = float(src_fps) if src_fps and src_fps > 1 else 30.0
    step, k, i = src_fps / fps, 0, 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            while int(round(k * step)) == i:
                yield i, img, src_fps
                k += 1
            i += 1
    finally:
        cap.release()


def extract(path, model, device, imgsz=640, conf=None, tracker="bytetrack.yaml"):
    kw = dict(tracker=tracker, verbose=False, device=device, imgsz=imgsz)
    if conf is not None:
        kw["conf"] = conf
    src_idx, luma = [], []
    d_frame, d_tid, d_kpts, d_box, d_conf = [], [], [], [], []
    w = h = 0
    src_fps = 30.0
    for f, (i, img, src_fps) in enumerate(read_resampled(path)):
        h, w = img.shape[:2]
        src_idx.append(i)
        luma.append(float(img[::8, ::8].mean()))
        r = model.track(img, persist=(f > 0), **kw)[0]      # 첫 프레임에서 추적기를 새로 만든다
        if r.boxes is None or r.boxes.id is None or r.keypoints is None or r.keypoints.conf is None:
            continue
        xy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy()
        bx = r.boxes.xyxy.cpu().numpy()
        bc = r.boxes.conf.cpu().numpy()
        for j, t in enumerate(r.boxes.id.int().cpu().tolist()):
            d_frame.append(f)
            d_tid.append(t)
            d_kpts.append(np.concatenate([xy[j], kc[j][:, None]], 1))
            d_box.append(bx[j])
            d_conf.append(bc[j])
    M = len(d_frame)
    return dict(
        n_frames=np.int32(len(src_idx)), fps=np.float32(FPS), src_fps=np.float32(src_fps),
        src_idx=np.array(src_idx, np.int32), width=np.int32(w), height=np.int32(h),
        luma=np.array(luma, np.float32),
        det_frame=np.array(d_frame, np.int32), det_tid=np.array(d_tid, np.int32),
        det_kpts=np.array(d_kpts, np.float32).reshape(M, 17, 3),
        det_box=np.array(d_box, np.float32).reshape(M, 4),
        det_conf=np.array(d_conf, np.float32),
    )


def pick_device(arg):
    if arg != "auto":
        return arg
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"   # MPS는 Ultralytics pose에서 알려진 오류가 있어 쓰지 않음


def main():
    ap = argparse.ArgumentParser(description="YOLO pose + ByteTrack 관절 추출 (모든 트랙 저장)")
    ap.add_argument("src", nargs="?", help="영상 파일 또는 폴더 (--dataset을 쓰면 생략)")
    ap.add_argument("--dataset", help="GMDCSA24 | caucafall | of-syn | OOPS")
    ap.add_argument("--out", help="출력 폴더 (기본 out/pose/<dataset>)")
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=None, help="검출 최소 신뢰도 (기본: Ultralytics 기본값)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N개만 (시험용)")
    ap.add_argument("--shuffle", action="store_true", help="영상을 고정된 무작위 순서로 처리")
    ap.add_argument("--shard", default="0/1", help="i/n: 영상을 n묶음으로 나눈 것 중 i번째만 (여러 터미널에서 동시에 돌릴 때)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.dataset:
        root = DATA / a.dataset / "video"
        out = Path(a.out or f"out/pose/{a.dataset}")
    else:
        root = Path(a.src)
        out = Path(a.out or "out/pose/custom")
    vids = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in VID_EXT)
    if a.shuffle:                                   # 중간에 멈춰도 그때까지의 결과가 전체의 무작위 표본이 되게
        vids = [vids[i] for i in np.random.default_rng(0).permutation(len(vids))]
    si, sn = map(int, a.shard.split("/"))
    vids = vids[si::sn]
    if a.limit:
        vids = vids[: a.limit]
    if not vids:
        raise SystemExit(f"영상이 없음: {root}")
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(a.model)
    device = pick_device(a.device)
    print(f"영상 {len(vids)}개, 모델 {a.model}, 장치 {device} -> {out}")
    log_path = out / ("extract_log.csv" if sn == 1 else f"extract_log_{si}of{sn}.csv")
    log_new = not log_path.exists()
    t_all, n_all = time.time(), 0
    with open(log_path, "a", newline="") as lf:
        wr = csv.writer(lf)
        if log_new:
            wr.writerow(["name", "n_frames", "n_det", "n_tracks", "sec", "status"])
        for k, p in enumerate(vids, 1):
            name = pose_name(p.relative_to(root)) if root.is_dir() else p.stem
            dst = out / f"{name}.npz"
            if dst.exists() and not a.overwrite:
                continue
            t0 = time.time()
            try:
                d = extract(p, model, device, a.imgsz, a.conf, a.tracker)
                np.savez_compressed(dst, model=a.model, imgsz=a.imgsz, tracker=a.tracker, **d)
                n = int(d["n_frames"])
                n_all += n
                wr.writerow([name, n, len(d["det_frame"]), len(np.unique(d["det_tid"])),
                             round(time.time() - t0, 1), "ok"])
                if k % 10 == 0 or k == len(vids):
                    el = time.time() - t_all
                    print(f"[{k}/{len(vids)}] {name}: {n}프레임 | 누적 {n_all / max(el, 1e-6):.1f} 프레임/초", flush=True)
            except Exception as e:                          # 한 영상이 실패해도 나머지는 계속
                print(f"[{k}/{len(vids)}] 실패 {name}: {e!r}", flush=True)
                wr.writerow([name, 0, 0, 0, round(time.time() - t0, 1), f"error: {e!r}"])
            lf.flush()


if __name__ == "__main__":
    main()
