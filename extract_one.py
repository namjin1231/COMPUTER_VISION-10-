import sys
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

def extract_folder(folder, step=2):
    pose = YOLO("yolov8n-pose.pt")          # 영상마다 새로 만들어야 추적 ID가 안 섞여
    files = sorted(Path(folder).rglob("*.png"))[::step]   # 30fps -> 15fps
    tracks = {}   # track_id -> [(frame_idx, kpts[17,3], box[4]), ...]
    for f, p in enumerate(files):
        img = cv2.imread(str(p))
        r = pose.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is None:
            continue
        xy = r.keypoints.xy.cpu().numpy()
        cf = r.keypoints.conf.cpu().numpy()
        for i, tid in enumerate(r.boxes.id.int().tolist()):
            k = np.concatenate([xy[i], cf[i][:, None]], axis=1)
            tracks.setdefault(tid, []).append((f, k, r.boxes.xyxy[i].cpu().numpy()))
    return tracks, len(files)

if __name__ == "__main__":
    folder, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks, n = extract_folder(folder)
    print(f"{folder.name}: 사용 프레임 {n}장, 원래 트랙 {len(tracks)}개")

    # UR Fall은 영상당 한 명이라 모든 트랙을 프레임 순서로 합친다
    merged = {}
    for items in tracks.values():
        for f, k, b in items:
            merged.setdefault(f, (k, b))     # 같은 프레임이 겹치면 먼저 나온 것 사용
    fs = sorted(merged)
    frames = np.array(fs)
    kpts = np.stack([merged[f][0] for f in fs])    # (T, 17, 3)
    boxes = np.stack([merged[f][1] for f in fs])   # (T, 4)
    np.savez(out_dir / f"{folder.name}.npz", frames=frames, kpts=kpts, boxes=boxes)
    print(f"  합친 결과: {len(fs)}프레임, kpts {kpts.shape}, 놓친 프레임 {n - len(fs)}장")