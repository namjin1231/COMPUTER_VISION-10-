"""데이터 받기 (제안서 5장). omnifall 패키지로 하나씩 받고 검증한다.

  python download_data.py labels                 # 라벨·공식 분할 CSV (수 MB)
  python download_data.py GMDCSA24               # 1.0 GiB
  python download_data.py of-syn caucafall       # 9.0 GiB, 7.8 GiB (caucafall은 ffmpeg 변환 필요)
  python download_data.py OOPS                   # OF-ItW. 전송 44.6 GiB, 저장 약 2.6 GiB
  python download_data.py night                  # 밤 영상 목록 (GMDCSA24 원본 CSV를 받은 뒤)

결과
  data/omnifall/<이름>/video/<path>.mp4
  data/omnifall/labels/<config>_<split>.csv      path, label(0~15), start, end, subject, cam, dataset
  data/omnifall/labels/<이름>_night.txt          밤 또는 불 끈 영상의 path 목록
  split.json                                     데이터셋별 학습/검증/테스트 영상 목록 (공식 분할)
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA = Path("data/omnifall")
VIDEO = ["GMDCSA24", "of-syn", "caucafall", "OOPS"]
LABEL_CONFIGS = ["gmdcsa24-cs", "caucafall-cs", "of-syn", "of-itw"]
CAUCAFALL_NIGHT_SUBJECTS = {6, 8}


def cli(args, env=None):
    exe = shutil.which("omnifall")
    cmd = ([exe] if exe else [sys.executable, "-m", "omnifall.cli"]) + args
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)


def get_video(name, workers=None):
    vdir = DATA / name / "video"
    vdir.mkdir(parents=True, exist_ok=True)          # omnifall은 폴더가 미리 있어야 한다
    env = dict(os.environ)
    if not shutil.which("ffmpeg") and "OMNIFALL_FFMPEG" not in env:
        try:
            import imageio_ffmpeg
            env["OMNIFALL_FFMPEG"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    base = ["--video-dir", str(vdir)]
    w = ["--workers", str(workers)] if workers else []
    rc = cli(["prepare", name, "--yes", "--download-dir", str(DATA / "_downloads")] + base + w, env)
    return rc or cli(["verify", name] + base, env)


def get_labels():
    import omnifall
    out = DATA / "labels"
    out.mkdir(parents=True, exist_ok=True)
    for cfg in LABEL_CONFIGS:
        parts = omnifall.load(cfg)
        for split in parts:
            parts[split].to_csv(str(out / f"{cfg}_{split}.csv"), index=False)
            print(f"{cfg}/{split}: 구간 {len(parts[split])}개, 영상 {len(set(parts[split]['path']))}개")
    write_split()


def write_split():
    """공식 분할의 영상 목록을 split.json으로 고정 (제안서 10장: 데이터 누수 대응)"""
    import json
    from dataset import CONFIGS, load_segments
    split = {}
    for ds in CONFIGS:
        split[ds] = {"train": [], "val": [], "test": []}
        for path, (sp, _) in sorted(load_segments(ds).items()):
            split[ds][sp].append(path)
    json.dump(split, open("split.json", "w"), indent=0)
    print("split.json:", {ds: {k: len(v) for k, v in s.items()} for ds, s in split.items()})


def get_night():
    out = DATA / "labels"
    # CAUCAFall: 피험자 6, 8이 야간 촬영
    paths = set()
    for f in out.glob("caucafall-cs_*.csv"):
        paths |= {r["path"] for r in csv.DictReader(open(f)) if int(r["subject"]) in CAUCAFALL_NIGHT_SUBJECTS}
    (out / "caucafall_night.txt").write_text("\n".join(sorted(paths)))
    print(f"caucafall 밤 영상 {len(paths)}개")
    # GMDCSA24: 원본 CSV의 'Time of Recording'이 Night 또는 Lights Off
    roots = list((DATA / "_downloads" / "GMDCSA24").rglob("Subject 1"))
    if not roots:
        print("GMDCSA24 원본 CSV가 없음 (data/omnifall/_downloads/GMDCSA24). 먼저 GMDCSA24를 받으세요.")
        return
    paths = set()
    for f in sorted(roots[0].parent.glob("Subject */*.csv")):
        subj = f.parent.name.replace(" ", "_")
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            when = (r.get("Time of Recording") or "").strip()
            if "Night" in when or "Off" in when:
                paths.add(f"{subj}/{f.stem}/{Path(r['File Name'].strip()).stem}")
    (out / "GMDCSA24_night.txt").write_text("\n".join(sorted(paths)))
    print(f"GMDCSA24 밤·불 끈 영상 {len(paths)}개")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", choices=VIDEO + ["labels", "night"])
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    failed = []
    for t in a.targets:
        if t == "labels":
            get_labels()
        elif t == "night":
            get_night()
        elif get_video(t, a.workers):
            failed.append(t)
    if failed:
        sys.exit(f"실패: {failed}")
