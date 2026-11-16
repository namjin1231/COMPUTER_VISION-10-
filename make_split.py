"""OmniFall 공식 분할로 split.json, 영상별 낮·밤으로 daynight.json을 만든다 (제안서 5장 '라벨과 분할', ⑦ 야간 대응)

영상 키 = "{dataset}/{path}"  (예: "GMDCSA24/Subject_1/Fall/01", "of-syn/fall/fall_to_001")
  추출 결과 파일 raw/{dataset}/{path}.npz 와 같은 이름이라 바로 연결된다

분할 (OmniFall config 그대로):
  of-syn        -> of-syn         train / validation / test (무작위 80/10/10)
  GMDCSA24      -> gmdcsa24-cs    피험자 1,3 / 2 / 4
  caucafall     -> caucafall-cs   피험자 1~7 / 10 / 8,9
  OOPS (OF-ItW) -> of-itw         test만 사용 (학습·검증에 안 씀)

낮·밤:
  GMDCSA24  원본 CSV의 'Time of Recording'. Night, Lights Off -> night, 나머지 -> day
  caucafall 피험자 6, 8 -> night (제안서), 나머지 -> day
  of-syn, OOPS -> unknown (낮밤 정보 없음, 야간 평가에 안 씀)

사용: python make_split.py [GMDCSA24 원본 폴더]
  원본 폴더 기본값: ~/.cache/omnifall/downloads/GMDCSA24/unpacked/GMDCSA24-...-master
"""
import csv
import json
import sys
from pathlib import Path

CONFIGS = [("of-syn", None), ("gmdcsa24-cs", None), ("caucafall-cs", None), ("of-itw", ["test"])]
CAUCA_NIGHT_SUBJECTS = {6, 8}
GMD_SRC = Path.home() / ".cache/omnifall/downloads/GMDCSA24/unpacked/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master"

def gmdcsa24_daynight(src):
    """원본 CSV -> {"GMDCSA24/Subject_1/Fall/01": "day" | "night"}"""
    out = {}
    for f in sorted(Path(src).glob("Subject */*.csv")):
        subj = f.parent.name.replace(" ", "_")          # 'Subject 1' -> 'Subject_1' (OmniFall 표기)
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            t = r["Time of Recording"].strip().lower()
            name = Path(r["File Name"].strip()).stem
            out[f"GMDCSA24/{subj}/{f.stem}/{name}"] = "night" if ("night" in t or "off" in t) else "day"
    return out

def video_subjects(config):
    """{영상 키: 피험자 번호}"""
    import omnifall
    out = {}
    for d in omnifall.load(config).values():
        for p, ds, s in zip(d["path"], d["dataset"], d["subject"]):
            out[f"{ds}/{p}"] = int(s)
    return out

if __name__ == "__main__":
    import omnifall
    gmd_src = Path(sys.argv[1]) if len(sys.argv) > 1 else GMD_SRC
    split = {"train": [], "validation": [], "test": []}
    daynight = {}
    gmd = gmdcsa24_daynight(gmd_src)
    if not gmd:
        sys.exit(f"GMDCSA24 원본 CSV를 못 찾음: {gmd_src}")

    for cfg, only in CONFIGS:
        ds = omnifall.load(cfg)
        subj = video_subjects(cfg)
        for sp, d in ds.items():
            if only and sp not in only:
                continue
            keys = sorted({f"{x}/{p}" for x, p in zip(d["dataset"], d["path"])})
            split[sp] += keys
            for k in keys:
                if k.startswith("GMDCSA24/"):
                    daynight[k] = gmd[k]                    # 없으면 KeyError로 바로 알 수 있게
                elif k.startswith("caucafall/"):
                    daynight[k] = "night" if subj[k] in CAUCA_NIGHT_SUBJECTS else "day"
                else:
                    daynight[k] = "unknown"

    for sp in split:
        split[sp].sort()
    # 한 영상이 두 분할에 들어가면 누수 -> 바로 멈춘다
    seen = {}
    for sp, keys in split.items():
        for k in keys:
            assert k not in seen, f"분할 겹침: {k} ({seen[k]}, {sp})"
            seen[k] = sp

    Path("split.json").write_text(json.dumps(split, indent=1, ensure_ascii=False))
    Path("daynight.json").write_text(json.dumps(dict(sorted(daynight.items())), indent=1, ensure_ascii=False))
    print({sp: len(v) for sp, v in split.items()}, "-> split.json, daynight.json")
