import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from lifelines import KaplanMeierFitter

# -------------------------------
# Font (Chinese-safe)
# -------------------------------
plt.rcParams["font.sans-serif"] = ["PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------------------------------
# 1. Configuration
# -----------------------------------------------------
target_words = [
    "机关", "激烈", "交通", "教授", "剧烈",
    "输入", "系统", "严格", "严厉", "严重",
    "组织", "作业"
]

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
EPSILON = 0.003

START_YEAR = 1931
DECADE_STEP = 10

# -----------------------------------------------------
# 2. Core functions
# -----------------------------------------------------
def compute_time_to_stabilization(decade_data, epsilon=EPSILON):
    jsds = [v for _, v in decade_data]
    for i in range(1, len(jsds) - 1):
        if jsds[i] is not None and jsds[i + 1] is not None:
            if jsds[i] < epsilon and jsds[i + 1] < epsilon:
                return i + 1, 1
    return len(jsds), 0


def build_survival_dataframe(data, corpus_label, word):
    decades = sorted(data.keys())
    senses = set()
    for d in data.values():
        senses.update(d.keys())

    records = []
    for sense in senses:
        trajectory = []
        for dec in decades:
            jsd = data.get(dec, {}).get(sense, {}).get("JSD", None)
            trajectory.append((dec, jsd))

        duration, event = compute_time_to_stabilization(trajectory)
        records.append({
            "Word": word,
            "Sense": sense,
            "duration": duration,
            "event": event,
            "Corpus": corpus_label
        })

    return pd.DataFrame(records)

# -----------------------------------------------------
# 3. Load data
# -----------------------------------------------------
dfs = []

for word in target_words:
    with open(RESULT_DIR / f"new_{word}_hongse_sense_shift_results.json", encoding="utf-8") as f:
        hongse_data = json.load(f)
    with open(RESULT_DIR / f"new_{word}_shenbao_sense_shift_results.json", encoding="utf-8") as f:
        shenbao_data = json.load(f)

    dfs.append(build_survival_dataframe(hongse_data, "Hongse", word))
    dfs.append(build_survival_dataframe(shenbao_data, "Shenbao", word))

df = pd.concat(dfs, ignore_index=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_path = TABLES_DIR / "survival_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
# # -----------------------------------------------------
# # 4. Matplotlib plot (inspection)
# # -----------------------------------------------------
# fig, ax = plt.subplots(figsize=(11, 7))

# reds = get_cmap("Reds")
# blues = get_cmap("Blues")

# n_words = len(target_words)
# levels = np.linspace(0.15, 0.95, n_words)

# legend_hongse, legend_shenbao = [], []

# # Store KM coordinates for TikZ
# tikz_data = []

# for i, word in enumerate(target_words):
#     red_color = reds(levels[i])
#     blue_color = blues(levels[i])

#     # ---- Hongse ----
#     subset_h = df[(df["Word"] == word) & (df["Corpus"] == "Hongse")]
#     kmf = KaplanMeierFitter()
#     kmf.fit(subset_h["duration"], subset_h["event"])
#     kmf.plot_survival_function(ax=ax, ci_show=False, color=red_color, linewidth=1.3)

#     legend_hongse.append(Line2D([0], [0], color=red_color, lw=2, label=word))
#     tikz_data.append(("Hongse", word, i, kmf.survival_function_))

#     # ---- Shenbao ----
#     subset_s = df[(df["Word"] == word) & (df["Corpus"] == "Shenbao")]
#     kmf = KaplanMeierFitter()
#     kmf.fit(subset_s["duration"], subset_s["event"])
#     kmf.plot_survival_function(ax=ax, ci_show=False, color=blue_color, linewidth=1.3)

#     legend_shenbao.append(Line2D([0], [0], color=blue_color, lw=2, label=word))
#     tikz_data.append(("Shenbao", word, i, kmf.survival_function_))

# # ---- Legend (two semantic columns)
# handles = legend_hongse + legend_shenbao
# labels = [f"{w}（红色）" for w in target_words] + [f"{w}（申报）" for w in target_words]

# ax.legend(
#     handles=handles,
#     labels=labels,
#     ncol=2,
#     title="Hongse (left)        Shenbao (right)",
#     frameon=False,
#     fontsize=9,
#     columnspacing=2.5
# )

# # ---- Decade labels
# max_duration = int(df["duration"].max())
# xticks = np.arange(1, max_duration + 1)
# xtick_labels = [
#     f"{START_YEAR + (i-1)*DECADE_STEP}–{START_YEAR + i*DECADE_STEP - 1}"
#     for i in xticks
# ]

# ax.set_xticks(xticks)
# ax.set_xticklabels(xtick_labels, rotation=40, ha="right")
# ax.set_xlabel("Decades")
# ax.set_ylabel("Survival probability (unstable)")
# ax.set_title("Sense Stabilization Trajectories (12 Words × 2 Corpora)")

# plt.tight_layout()
# plt.show()

# # -----------------------------------------------------
# # 5. Direct PGFPlots / TikZ export (NO tikzplotlib)
# # -----------------------------------------------------
# def pgf_color(corpus, i, n):
#     level = int(30 + 60 * (0.15 + 0.8 * i / (n - 1)))
#     return f"red!{level}" if corpus == "Hongse" else f"blue!{level}"

# def decade_label(t):
#     start = START_YEAR + (t - 1) * DECADE_STEP
#     end = start + 9
#     return f"{start}-{end}"

# with open("sense_stabilization_km.tex", "w", encoding="utf-8") as f:
#     f.write(r"""
# \begin{tikzpicture}
# \begin{axis}[
#     width=12cm,
#     height=8cm,
#     xlabel={Decades},
#     ylabel={Survival probability (unstable)},
#     ymin=0, ymax=1,
#     xtick={""" + ",".join(map(str, xticks)) + r"""},
#     xticklabels={""" + ",".join(decade_label(i) for i in xticks) + r"""},
#     legend columns=2,
#     legend style={
#         at={(0.5,-0.25)},
#         anchor=north,
#         draw=none,
#         column sep=2em
#     }
# ]
# """)

#     for corpus, word, i, sf in tikz_data:
#         color = pgf_color(corpus, i, n_words)
#         f.write(f"\n% {word} {corpus}\n")
#         f.write(f"\\addplot+[thick, {color}, const plot]\ncoordinates {{\n")
#         for t, s in zip(sf.index, sf.iloc[:, 0]):
#             f.write(f"({int(t)},{float(s):.4f})\n")
#         f.write("};\n")
#         f.write(f"\\addlegendentry{{{word}（{'红色' if corpus=='Hongse' else '申报'}）}}\n")

#     f.write(r"""
# \end{axis}
# \end{tikzpicture}
# """)
