"""평가 (제안서 6장, 전체 테스트 분할 행): 실시간 방식으로 예측한 프레임별 클래스별 F1, macro-F1, 혼동행렬

  python evaluate.py --run out/runs/e1_gru --datasets GMDCSA24 --split val
  python evaluate.py --run out/runs/e1_gru --datasets GMDCSA24 --split test     # 테스트는 마지막에 한 번만

결과는 화면과 <run>/eval_<split>.json. 데이터셋별, 낮·밤별로도 나눠서 낸다.
정답이 제외(-1)인 프레임은 점수에서 뺀다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import CLASS_NAMES, EXCLUDE, NCLS, Corpus
from infer import predict_corpus
from models import load_ckpt


def confusion(y, pred):
    m = np.zeros((NCLS, NCLS), np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def f1_scores(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def frame_metrics(y, pred):
    """제외 프레임을 빼고 계산. macro-F1은 정답에 한 번이라도 나온 클래스만 평균한다
    (예: 일상 영상뿐인 밤 영상 묶음에 낙상 클래스가 없을 때 0점으로 깎이지 않게)."""
    keep = y != EXCLUDE
    cm = confusion(y[keep], pred[keep])
    f1 = f1_scores(cm)
    present = cm.sum(1) > 0
    return dict(n_frames=int(keep.sum()), accuracy=float(np.trace(cm) / max(cm.sum(), 1)),
                macro_f1=float(f1[present].mean()) if present.any() else 0.0,
                f1=[round(float(v), 4) for v in f1], support=cm.sum(1).tolist(), confusion=cm.tolist())


def macro_f1(y, pred):
    return frame_metrics(y, pred)["macro_f1"]


def evaluate(model, corpus):
    pred = predict_corpus(model, corpus).argmax(1)
    vid = np.repeat(np.arange(len(corpus)), np.diff(corpus.start))
    ds = np.array(corpus.dataset)[vid]
    night = np.array(corpus.night, bool)[vid]
    out = dict(all=dict(frame_metrics(corpus.y, pred), n_videos=len(corpus)))
    groups = [(f"{d}", ds == d, np.array(corpus.dataset) == d) for d in sorted(set(corpus.dataset))]
    groups += [("낮", ~night, ~np.array(corpus.night, bool)), ("밤", night, np.array(corpus.night, bool))]
    for name, fm, vm in groups:
        if fm.any():
            out[name] = dict(frame_metrics(corpus.y[fm], pred[fm]), n_videos=int(vm.sum()))
    keep = corpus.y != EXCLUDE
    out["all"]["miss_rate"] = float(1 - corpus.det[keep].mean()) if keep.any() else 0.0   # 피험자를 못 잡은 프레임 비율
    return out


def show(res, title):
    print(f"== {title}")
    for name, m in res.items():
        print(f"[{name}] 영상 {m['n_videos']}개, 프레임 {m['n_frames']} | macro-F1 {m['macro_f1']:.3f}, "
              f"정확도 {m['accuracy']:.3f} | F1 {CLASS_NAMES} = {m['f1']}")
    print("혼동행렬 (전체, 행 = 정답, 열 = 예측):")
    print(np.array(res["all"]["confusion"]))
    print(f"피험자 미검출 프레임 비율 {res['all']['miss_rate']:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train.py의 출력 폴더")
    ap.add_argument("--data", default="out/data")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    run = Path(a.run)
    corpus = Corpus(a.data, a.datasets, a.split)
    model, ck = load_ckpt(run / "best.pt")
    res = evaluate(model, corpus)
    show(res, f"{run.name} best.pt ({ck['model']}, seed {ck.get('seed')}) on {a.split}")
    seeds = {}
    for p in sorted(run.glob("seed*.pt")):
        seeds[p.stem] = evaluate(load_ckpt(p)[0], corpus)["all"]["macro_f1"]
    if seeds:
        v = np.array(list(seeds.values()))
        print(f"시드 {len(v)}개 macro-F1 {v.mean():.3f} ± {v.std():.3f}  {dict((k, round(x, 3)) for k, x in seeds.items())}")
    json.dump(dict(split=a.split, datasets=a.datasets, best=res, seeds=seeds),
              open(run / f"eval_{a.split}.json", "w"), ensure_ascii=False, indent=1)
