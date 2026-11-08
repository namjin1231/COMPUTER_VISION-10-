import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from extract_one import extract_folder
from features import compute_features
from dataset import load_labels, frame_labels, make_windows, NCLS

ROOT = Path("data/urfall")
CSV = {"fall": ROOT / "urfall-cam0-falls.csv", "adl": ROOT / "urfall-cam0-adls.csv"}
RAW, FEAT, WIN = Path("out/raw"), Path("out/feat"), Path("out/win")

def unzip_all(kind):
    """zip이 있으면 같은 이름의 폴더로 풀기 (이미 풀려 있으면 건너뜀)
    _tmp_ 폴더에 다 푼 뒤 이름을 바꾸므로, 중간에 끊겨도 덜 풀린 폴더가 '완료'로 남지 않는다.
    zip 하나가 깨져 있어도 경고만 찍고 다음 zip으로 넘어간다."""
    for z in sorted((ROOT / kind).glob("*.zip")):
        dest = ROOT / kind / z.stem
        if dest.exists():
            continue
        tmp = ROOT / kind / f"_tmp_{z.stem}"
        shutil.rmtree(tmp, ignore_errors=True)       # 지난번에 끊긴 찌꺼기 제거
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"압축 풀기 실패 (zip이 깨졌을 수 있음, 다시 받기 필요): {z.name}: {e!r}")
            continue
        for i in range(10):                            # Windows: 백신/인덱서가 방금 푼 파일을 잠깐 잡고 있으면
            try:                                       # 이름 바꾸기가 거부되므로 잠시 후 다시 시도
                tmp.rename(dest)
                break
            except PermissionError:
                time.sleep(1 + i)
        else:
            print(f"폴더 이름 바꾸기 실패, 다음 실행 때 다시 풂: {tmp.name}")
            continue
        print("압축 풀기:", z.name)

def merge_tracks(tracks):
    """한 영상의 트랙을 프레임 순서로 하나로 합치기 (UR Fall은 영상당 한 명)"""
    merged = {}
    for items in tracks.values():
        for f, k, b in items:
            merged.setdefault(f, (k, b))
    fs = sorted(merged)
    return (np.array(fs),
            np.stack([merged[f][0] for f in fs]),
            np.stack([merged[f][1] for f in fs]))

def process(folder, kind):
    name = folder.name                    # fall-01-cam0-rgb
    seq = name.split("-cam0")[0]          # fall-01
    if (WIN / f"{seq}.npz").exists():
        print("건너뜀(이미 있음):", seq)
        return
    tracks, n = extract_folder(folder)
    if not tracks:
        print(f"사람을 못 찾음: {seq}")
        return
    frames, kpts, boxes = merge_tracks(tracks)
    for d in (RAW, FEAT, WIN):
        d.mkdir(parents=True, exist_ok=True)
    np.savez(RAW / f"{name}.npz", frames=frames, kpts=kpts, boxes=boxes, n=n)   # n = 전체 프레임 수(15fps)
    feat = compute_features(RAW / f"{name}.npz")
    np.save(FEAT / f"{name}.npy", feat.astype(np.float32))
    y = frame_labels(load_labels(CSV[kind], seq), len(feat), kind)
    X, Y, S = make_windows(feat, y)
    np.savez(WIN / f"{seq}.npz", X=X.astype(np.float32), y=Y,
             start=S, vid=np.array([seq] * len(Y)))
    print(f"{seq}: 프레임 {n}장, 원래 트랙 {len(tracks)}개, 창 {len(Y)}개, "
          f"클래스 {np.bincount(Y, minlength=NCLS).tolist()}")

if __name__ == "__main__":
    for kind in ("fall", "adl"):
        unzip_all(kind)
        for folder in sorted(p for p in (ROOT / kind).iterdir()
                             if p.is_dir() and not p.name.startswith("_tmp_")):
            try:
                process(folder, kind)
            except Exception as e:           # 한 영상이 실패해도 나머지는 계속
                print(f"실패 {folder.name}: {e!r}")