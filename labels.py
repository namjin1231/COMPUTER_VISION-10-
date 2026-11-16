"""OmniFall 시간 구간 라벨 -> 15fps 프레임별 3클래스 정답 (제안서 5장 '라벨과 분할', 4.3 ③)

3클래스: 0 정상 / 1 넘어지는 중(fall) / 2 쓰러짐(fallen)
- lying(가만히 누워 있음)은 쓰러진 자세와 관절 모양이 같아서 제외(-1). 손실에서 뺀다
- lie_down(천천히 눕기)을 포함한 나머지 13개 라벨은 전부 정상(0)
- 어느 구간에도 안 들어가는 시각도 제외(-1)
- 구간이 겹치면 나중에 시작한 구간을 따른다 (다음 동작이 이어받음. 예: fall/fallen이 겹치면 fallen)
  OmniFall 구간은 대부분 0.0x초 반올림 차이로 겹치거나 비어 있다

사용:
  from labels import load_segments, frame_labels
  seg = load_segments("gmdcsa24-cs")              # {path: [(start, end, 라벨 이름), ...]}
  y = frame_labels(seg["Subject_1/Fall/01"], n)   # (n,) int, 15fps 프레임 k의 시각 k/15초 기준
"""
import collections
import numpy as np

FPS = 15
NCLS = 3
EXCLUDE = -1
CLASS_NAMES = ["정상", "넘어지는 중", "쓰러짐"]
OMNIFALL_TO_3 = {"fall": 1, "fallen": 2, "lying": EXCLUDE}   # 나머지는 전부 0

def to3(name):
    return OMNIFALL_TO_3.get(name, 0)

def load_segments(config):
    """OmniFall config의 모든 분할에서 영상별 구간 목록을 모은다. 영상별로 시작 시각 순 정렬.
    반환: {path: [(start, end, 라벨 이름), ...]}, {path: 분할 이름}"""
    import omnifall
    ds = omnifall.load(config)
    seg, split = collections.defaultdict(list), {}
    for sp, d in ds.items():
        names = d.features["label"].names
        for p, a, b, l in zip(d["path"], d["start"], d["end"], d["label"]):
            seg[p].append((float(a), float(b), names[l]))
            split[p] = sp
    for p in seg:
        seg[p].sort()
    return dict(seg), split

def frame_labels(segments, n, fps=FPS):
    """구간 목록 -> (n,) 프레임별 정답. 프레임 k는 시각 k/fps.
    시작 순으로 덮어쓰므로, 겹치면 나중에 시작한 구간이 이긴다."""
    t = np.arange(n) / fps
    y = np.full(n, EXCLUDE, dtype=np.int64)
    for a, b, name in segments:               # 이미 시작 순 정렬
        y[(t >= a) & (t < b)] = to3(name)
    return y

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "gmdcsa24-cs"
    seg, split = load_segments(cfg)
    tot = collections.Counter()
    for p, s in seg.items():
        n = int(np.ceil(s[-1][1] * FPS))      # 라벨 끝까지 (실제 길이는 추출 결과의 프레임 수를 쓴다)
        tot.update(frame_labels(s, n).tolist())
    print(f"{cfg}: 영상 {len(seg)}개, 프레임별 정답 개수",
          {**{CLASS_NAMES[c]: tot[c] for c in range(NCLS)}, "제외": tot[EXCLUDE]})
