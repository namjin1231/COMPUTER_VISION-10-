"""학습 결과 그래프 만들기: out/runs/e2_*/의 평가 결과 -> docs/figures/*.png (맥 기본 한글 글꼴 사용)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")   # 실행: python docs/make_figures.py
outd = root / "docs" / "figures"
outd.mkdir(parents=True, exist_ok=True)

font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "text.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
})
INK2 = "#52514e"
MODELS = ["tcnte", "gru", "cnn"]
LABEL = {"tcnte": "TCNTE (최종)", "gru": "GRU", "cnn": "1D-CNN"}
COLOR = {"tcnte": "#2a78d6", "gru": "#eb6834", "cnn": "#1baf7a"}   # 검증한 categorical 1~3번, 고정 순서

R = {m: {t: json.load(open(root / f"out/runs/e2_{m}/eval_{t}.json")) for t in ("val", "test", "test_oops")} for m in MODELS}
S = {m: json.load(open(root / f"out/runs/e2_{m}/summary.json")) for m in MODELS}


def seeds(m, t):
    v = np.array(list(R[m][t]["seeds"].values()))
    return v.mean(), v.std()


def bars(ax, groups, getter, ylabel, ylim, fmt="{:.3f}", err=None):
    x = np.arange(len(groups))
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = [getter(m, g) for g in groups]
        errs = [err(m, g) for g in groups] if err else None
        pos = x + (i - 1) * (w + 0.01)
        ax.bar(pos, vals, w, color=COLOR[m], label=LABEL[m], edgecolor="#fcfcfb", linewidth=2,
               yerr=errs, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
        for p, v, e in zip(pos, vals, errs or [0] * len(vals)):
            ax.text(p, v + (e or 0) + 0.008, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3)


# 1. 평가 세트별 macro-F1 (시드 3개 평균 ± 표준편차)
fig, ax = plt.subplots(figsize=(9, 4.6))
groups = {"검증\n(1,208개)": "val", "테스트\n(OF-Syn·GMDCSA24·CAUCAFall, 1,256개)": "test", "테스트 OOPS\n(실제 사고 영상, 572개)": "test_oops"}
bars(ax, list(groups), lambda m, g: seeds(m, groups[g])[0], "macro-F1 (시드 3개 평균)", (0, 0.9),
     err=lambda m, g: seeds(m, groups[g])[1])
ax.set_title("평가 세트별 macro-F1 (오차 막대 = 시드 표준편차)", loc="left")
fig.tight_layout()
fig.savefig(outd / "1_macro_f1_by_split.png", dpi=150)

# 2. 클래스별 F1 (테스트, best.pt)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
cls = ["정상", "넘어지는 중", "쓰러짐"]
for ax, (t, title) in zip(axs, [("test", "테스트 (1,256개)"), ("test_oops", "테스트 OOPS (572개)")]):
    bars(ax, cls, lambda m, g: R[m][t]["best"]["all"]["f1"][cls.index(g)], "F1 (best.pt)", (0, 1.08))
    ax.set_title(f"클래스별 F1 · {title}", loc="left")
axs[1].set_ylabel("")
axs[1].get_legend().remove()
fig.tight_layout()
fig.savefig(outd / "2_f1_by_class.png", dpi=150)

# 3. 데이터셋·낮밤별 macro-F1 (테스트, best.pt)
fig, ax = plt.subplots(figsize=(10, 4.6))
sub = {"OF-Syn\n(1,200개)": ("test", "of-syn"), "GMDCSA24\n(37개)": ("test", "GMDCSA24"),
       "CAUCAFall\n(19개)": ("test", "caucafall"), "밤 영상\n(28개)": ("test", "밤"), "OOPS\n(572개)": ("test_oops", "OOPS")}
bars(ax, list(sub), lambda m, g: R[m][sub[g][0]]["best"][sub[g][1]]["macro_f1"], "macro-F1 (best.pt)", (0, 0.95))
ax.set_title("테스트 분할: 데이터셋·밤 영상별 macro-F1", loc="left")
fig.tight_layout()
fig.savefig(outd / "3_macro_f1_by_dataset.png", dpi=150)

# 4. 학습 곡선 (검증 macro-F1, 모델별 시드 3개)
fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
styles = ["-", "--", ":"]
for ax, m in zip(axs, MODELS):
    for r, ls in zip(S[m]["seeds"], styles):
        h = r["history"]
        ax.plot([e["epoch"] for e in h], [e["val_f1"] for e in h], ls, color=COLOR[m], lw=2, label=f"시드 {r['seed']}")
        ax.plot(r["epoch"], r["val_macro_f1"], "o", ms=8, color=COLOR[m], mec="#fcfcfb", mew=2)
    ax.set_title(f"{LABEL[m]} · {S[m]['val_macro_f1_mean']:.3f} ± {S[m]['val_macro_f1_std']:.3f}", loc="left")
    ax.set_xlabel("epoch")
    ax.legend(loc="lower right")
axs[0].set_ylabel("검증 macro-F1")
axs[0].set_ylim(0.4, 0.8)
fig.suptitle("학습 곡선 (점 = 시드별 최고 epoch, 이후 15 epoch 개선 없으면 조기 종료)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(outd / "4_learning_curves.png", dpi=150)

# 5. 혼동행렬 (TCNTE best.pt, 정답 행 기준 비율)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, (t, title) in zip(axs, [("test", "테스트 (1,256개)"), ("test_oops", "테스트 OOPS (572개)")]):
    cm = np.array(R["tcnte"][t]["best"]["all"]["confusion"], float)
    p = cm / cm.sum(1, keepdims=True)
    ax.imshow(p, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#0d366b"]), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{p[i, j]:.0%}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=9.5,
                    color="#fcfcfb" if p[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks(range(3), cls)
    ax.set_yticks(range(3), cls)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"TCNTE 혼동행렬 · {title}", loc="left")
fig.tight_layout()
fig.savefig(outd / "5_confusion_tcnte.png", dpi=150)
print("ok", sorted(p.name for p in outd.iterdir()))
