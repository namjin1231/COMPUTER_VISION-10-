"""③ 학습 샘플 만들기 (제안서 4.3 ③)

만들기:  python dataset.py --datasets GMDCSA24
  관절(out/pose) + OmniFall 구간 라벨(data/omnifall/labels) -> 피험자 선택 -> 특징 -> 프레임별 정답
  -> out/data/<dataset>.npz, out/data/<dataset>_videos.csv(영상별 선택 결과)

정답 (프레임별, 3클래스)
  fall -> 1 넘어지는 중, fallen -> 2 쓰러짐, lying -> -1 제외, 그 밖의 라벨 -> 0 정상
  어떤 구간에도 속하지 않는 프레임은 정답을 모르므로 -1
창
  30프레임(2초), 5프레임 간격, 창 끝 프레임 e = 4, 9, 14, ...
  30프레임이 안 찬 앞부분은 0으로 채우고 정답은 -1 (손실에서 빠짐)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from extract_pose import FPS, pose_name
from features import N_FEAT, compute_features
from select_subject import select_subject

WIN, STRIDE = 30, 5
NCLS = 3
CLASS_NAMES = ["정상", "넘어지는 중", "쓰러짐"]
EXCLUDE = -1
FALL, FALLEN, LYING = 1, 2, 6                    # OmniFall 라벨 번호
LABELS = Path("data/omnifall/labels")
# 데이터셋 -> (라벨 config, 쓰는 분할). OF-ItW는 테스트 전용
CONFIGS = {
    "GMDCSA24": ("gmdcsa24-cs", ("train", "validation", "test")),
    "caucafall": ("caucafall-cs", ("train", "validation", "test")),
    "of-syn": ("of-syn", ("train", "validation", "test")),
    "OOPS": ("of-itw", ("test",)),
}
SYNTHETIC = {"of-syn"}
SPLIT_NAME = {"train": "train", "validation": "val", "test": "test"}


def map_label(omni):
    return {FALL: 1, FALLEN: 2, LYING: EXCLUDE}.get(int(omni), 0)


def frame_labels(segments, n_frames, fps=FPS):
    """segments: [(시작 초, 끝 초, OmniFall 라벨)] -> (n_frames,) 프레임 k의 시각은 k / fps"""
    y = np.full(n_frames, EXCLUDE, np.int8)
    t = np.arange(n_frames) / fps
    for start, end, lab in sorted(segments):
        y[(t >= start) & (t < end)] = map_label(lab)
    return y


def load_segments(dataset):
    """{영상 경로: (분할, [(시작, 끝, 라벨)])}"""
    cfg, splits = CONFIGS[dataset]
    out = {}
    for sp in splits:
        f = LABELS / f"{cfg}_{sp}.csv"
        if not f.exists():
            raise SystemExit(f"라벨 파일이 없음: {f}  (python download_data.py labels)")
        for r in csv.DictReader(open(f)):
            out.setdefault(r["path"], (SPLIT_NAME[sp], []))[1].append(
                (float(r["start"]), float(r["end"]), int(r["label"])))
    return out


def night_videos(dataset):
    """밤(또는 불 끈) 영상 경로 집합. GMDCSA24는 원본 CSV의 시간대, CAUCAFall은 피험자 6·8"""
    f = LABELS / f"{dataset}_night.txt"
    return set(f.read_text().split("\n")) if f.exists() else set()


def build(dataset, pose_dir, out_dir):
    segs = load_segments(dataset)
    night = night_videos(dataset)
    feats, ys, dets, starts = [], [], [], [0]
    names, splits, nights, rows = [], [], [], []
    for path in sorted(segs):
        split, seg = segs[path]
        row = dict(video=path, split=split, night=int(path in night), status="no_pose", coverage=0.0,
                   n_frames=0, n_tracks=0, n_people=0, fallen_frames=0, fallen_detected=0)
        rows.append(row)
        pf = Path(pose_dir) / f"{pose_name(path)}.npz"
        if not pf.exists():
            continue
        pose = np.load(pf)
        y = frame_labels(seg, int(pose["n_frames"]))
        sel = select_subject(pose, y, use_labels=split != "test")
        row.update(status=sel["status"], coverage=round(sel["coverage"], 3), n_frames=len(y),
                   n_tracks=sel["n_tracks"], n_people=sel["n_people"],
                   fallen_frames=int((y == 2).sum()), fallen_detected=int((sel["valid"] & (y == 2)).sum()))
        if sel["status"] != "ok":
            continue
        feats.append(compute_features(sel["kpts"], sel["box"], sel["valid"]))
        ys.append(y)
        dets.append(sel["valid"])
        starts.append(starts[-1] + len(y))
        names.append(path)
        splits.append(split)
        nights.append(path in night)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if names:
        np.savez_compressed(
            out_dir / f"{dataset}.npz",
            feat=np.concatenate(feats).astype(np.float32), y=np.concatenate(ys).astype(np.int8),
            det=np.concatenate(dets), vid_start=np.array(starts, np.int64),
            names=np.array(names), split=np.array(splits), night=np.array(nights, bool))
    with open(out_dir / f"{dataset}_videos.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    report(dataset, rows, np.concatenate(ys) if ys else np.zeros(0, int), splits, starts)


def report(dataset, rows, y, splits, starts):
    print(f"== {dataset}: 라벨 있는 영상 {len(rows)}개")
    for sp in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == sp]
        if not rs:
            continue
        st = {}
        for r in rs:
            st[r["status"]] = st.get(r["status"], 0) + 1
        idx = [i for i, s in enumerate(splits) if s == sp]
        yy = np.concatenate([y[starts[i]:starts[i + 1]] for i in idx]) if idx else np.zeros(0, int)
        cnt = [int((yy == c).sum()) for c in range(NCLS)]
        ff = sum(r["fallen_frames"] for r in rs if r["status"] != "no_pose")
        fd = sum(r["fallen_detected"] for r in rs if r["status"] != "no_pose")
        print(f"  {sp:5s} 영상 {len(rs)}개 {st} | 사용 {len(idx)}개, 제외 {1 - len(idx) / len(rs):.1%} | "
              f"프레임 {CLASS_NAMES} = {cnt}, 제외 프레임 {int((yy == EXCLUDE).sum())} | "
              f"쓰러짐 구간 검출률 {fd / ff:.1%}" if ff else
              f"  {sp:5s} 영상 {len(rs)}개 {st} | 사용 {len(idx)}개 | 프레임 {cnt}")


# ---------------------------------------------------------------- 학습·평가에서 읽기
class Corpus:
    """여러 데이터셋의 한 분할을 이어 붙인 것. 영상 v의 프레임은 feat[start[v]:start[v+1]]"""

    def __init__(self, data_dir, datasets, split):
        feats, ys, dets, self.names, self.dataset, self.night = [], [], [], [], [], []
        lens = []
        for ds in datasets:
            # npz는 d["feat"]처럼 꺼낼 때마다 압축을 새로 푼다. 영상마다 꺼내면 영상 수만큼 전체를 다시 풀므로 한 번만 꺼낸다
            d = dict(np.load(Path(data_dir) / f"{ds}.npz"))
            vs = d["vid_start"]
            for i in np.where(d["split"] == split)[0]:
                feats.append(d["feat"][vs[i]:vs[i + 1]])
                ys.append(d["y"][vs[i]:vs[i + 1]])
                dets.append(d["det"][vs[i]:vs[i + 1]])
                lens.append(vs[i + 1] - vs[i])
                self.names.append(str(d["names"][i]))
                self.dataset.append(ds)
                self.night.append(bool(d["night"][i]))
        self.feat = np.concatenate(feats) if feats else np.zeros((0, N_FEAT), np.float32)
        self.y = np.concatenate(ys).astype(np.int64) if ys else np.zeros(0, np.int64)
        self.det = np.concatenate(dets) if dets else np.zeros(0, bool)
        self.start = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    def __len__(self):
        return len(self.names)

    def video(self, v):
        s, e = self.start[v], self.start[v + 1]
        return self.feat[s:e], self.y[s:e]

    def windows(self):
        """창 목록 (영상 번호, 창 끝 프레임). 30프레임 모두 제외인 창은 뺀다."""
        vid, end = [], []
        for v in range(len(self)):
            y = self.y[self.start[v]:self.start[v + 1]]
            for e in range(STRIDE - 1, len(y), STRIDE):
                if (y[max(0, e - WIN + 1): e + 1] != EXCLUDE).any():
                    vid.append(v)
                    end.append(e)
        return np.array(vid, np.int64), np.array(end, np.int64)


def gather(corpus, vid, end, speed=None):
    """창 묶음 꺼내기 -> X (B, 30, 55), Y (B, 30).
    speed (B,)를 주면 창 끝을 고정하고 그 배속으로 과거를 다시 뽑는다 (속도 증강).
    영상 시작보다 앞은 X = 0, Y = -1."""
    B = len(vid)
    f = np.ones(B) if speed is None else np.asarray(speed, float)
    pos = end[:, None] - (WIN - 1 - np.arange(WIN))[None, :] * f[:, None]        # (B, 30) 영상 안 위치
    ok = pos >= 0
    p = np.clip(pos, 0, None)
    i0 = np.floor(p).astype(np.int64)
    i1 = np.minimum(i0 + 1, end[:, None])
    a = (p - i0)[:, :, None].astype(np.float32)
    base = corpus.start[vid][:, None]
    X = corpus.feat[base + i0] * (1 - a) + corpus.feat[base + i1] * a
    Y = corpus.y[base + np.minimum(np.round(p).astype(np.int64), end[:, None])]
    X[~ok] = 0.0
    Y = np.where(ok, Y, EXCLUDE)
    return X.astype(np.float32), Y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--pose", default="out/pose")
    ap.add_argument("--out", default="out/data")
    a = ap.parse_args()
    for ds in a.datasets:
        build(ds, Path(a.pose) / ds, a.out)
