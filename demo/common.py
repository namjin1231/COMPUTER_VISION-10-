"""공통 설정: 경로, 데이터셋 목록, 라벨 변환. 모든 스크립트가 import 한다."""
import os, sys, json, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent          # fall_kit 폴더 = 작업 폴더
RAW = ROOT / "raw"
LABELS = ROOT / "labels"
# 변형 실험: 환경변수 FALL_VARIANT=<값> 이면 pose_<값>, pose_sel_<값>, seq_<값>, runs_<값>, logs/variant_<값> 을 쓴다.
# 값이 없으면 원래 경로 그대로 (기존 결과를 덮어쓰지 않기 위함)
VARIANT = os.environ.get("FALL_VARIANT", "").strip()
_S = f"_{VARIANT}" if VARIANT else ""
POSE = ROOT / f"pose{_S}"
POSE_SEL = ROOT / f"pose_sel{_S}"
SEQ = ROOT / f"seq{_S}"
RUNS = ROOT / f"runs{_S}"
LOGS = ROOT / "logs" / f"variant_{VARIANT}" if VARIANT else ROOT / "logs"
RESULTS = ROOT / f"results{_S}.csv"             # 07/08이 모아 쓰는 결과표
for d in (RAW, LABELS, POSE, POSE_SEL, SEQ, RUNS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ffmpeg를 설치할 수 없으면 zip을 받아 fall_kit/ffmpeg/bin 에 풀어두면 자동으로 쓴다
_ff = ROOT / "ffmpeg" / "bin"
if _ff.exists():
    os.environ.setdefault("OMNIFALL_FFMPEG", str(_ff))
    os.environ["PATH"] = str(_ff) + os.pathsep + os.environ.get("PATH", "")

FPS = 15          # 모든 영상을 15fps로 맞춘다
W = 30            # 2초 창

# dataset 열 값 -> OmniFall config 이름, 쓰는 분할
DATASETS = {
    "of-syn":    {"config": "of-syn",       "splits": ["train", "validation", "test"]},
    "GMDCSA24":  {"config": "gmdcsa24-cs",  "splits": ["train", "validation", "test"]},
    "caucafall": {"config": "caucafall-cs", "splits": ["train", "validation", "test"]},
    "OOPS":      {"config": "of-itw",       "splits": ["test"]},          # 테스트 전용
}
STAGED = ("GMDCSA24", "caucafall")   # 실제 촬영 데이터 (학습 비중 맞추기 대상)

# OmniFall 데이터 카드 기준 구간 수 (3-1 확인용)
EXPECTED_SEGMENTS = {
    ("of-syn", "train"): 15344, ("of-syn", "val"): 1956, ("of-syn", "test"): 1928,
    ("GMDCSA24", "train"): 213, ("GMDCSA24", "val"): 152, ("GMDCSA24", "test"): 93,
    ("caucafall", "train"): 176, ("caucafall", "val"): 25, ("caucafall", "test"): 47,
    ("OOPS", "test"): 3673,
}

# 16클래스: 0 walk 1 fall 2 fallen 3 sit_down 4 sitting 5 lie_down 6 lying 7 stand_up 8 standing 9 other ...
FALL, FALLEN, LYING = 1, 2, 6
CLASS_NAMES = ["normal", "fall", "fallen"]


def to3(label16):
    """OmniFall 16클래스 -> 우리 3클래스. omnifall.to_fall_state는 번호 순서가 달라서 쓰지 않는다."""
    label16 = np.asarray(label16)
    y = np.zeros(label16.shape, np.int8)       # 0 = 정상
    y[label16 == FALL] = 1                     # 넘어지는 중
    y[label16 == FALLEN] = 2                   # 쓰러져 있음
    y[label16 == LYING] = -1                   # 누워 있는 상태 = 제외
    y[label16 < 0] = -1                        # 라벨 구간 밖 = 제외
    return y


def vid_id(path):
    return path.replace("/", "__")


def video_dir(dataset):
    return RAW / dataset


def free_gb(path=ROOT):
    return shutil.disk_usage(path).free / 1e9


def append_changes(text):
    with open(LOGS / "changes.md", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
