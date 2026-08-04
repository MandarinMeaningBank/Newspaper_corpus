import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
TEX_DIR = OUTPUTS_DIR / "tex"

survival_df = pd.read_csv(TABLES_DIR / "survival_data.csv")

def compute_weighted_change(
    sense_data,
    max_decade_index
):
    """
    sense_data: dict
        { decade_label -> {"JSD": ..., "entropy": ...} }

    max_decade_index: int
        duration from survival_data.csv

    returns:
        total_weighted_change (float)
        max_weighted_change (float)
        peak_decade (str or None)
    """
    decades = sorted(sense_data.keys())

    total_weighted_change = 0.0
    max_weighted_change = 0.0
    peak_decade = None

    # only consider periods before stabilization
    for i in range(1, min(max_decade_index, len(decades))):
        prev_dec = decades[i - 1]
        curr_dec = decades[i]

        jsd = sense_data[curr_dec].get("JSD", None)
        H_prev = sense_data[prev_dec].get("entropy", None)
        H_curr = sense_data[curr_dec].get("entropy", None)

        if jsd is None or np.isnan(jsd):
            continue
        if H_prev is None or H_curr is None:
            continue

        weighted_change = jsd * (H_prev + H_curr) / 2

        # ADD it (this is the key change)
        total_weighted_change += weighted_change

        # still track peak
        if weighted_change > max_weighted_change:
            max_weighted_change = weighted_change
            peak_decade = curr_dec

    average_weighted_change = total_weighted_change / max_decade_index

    return average_weighted_change, max_weighted_change, peak_decade



records = []

for _, row in survival_df.iterrows():
    word     = row["Word"]
    sense    = str(row["Sense"])
    corpus   = row["Corpus"]
    duration = int(row["duration"])
    event    = int(row["event"])

    path = RESULT_DIR / f"new_{word}_{corpus.lower()}_sense_shift_results.json"
    with open(path, encoding="utf-8") as f:
        word_data = json.load(f)

    sense_data = {}
    for dec, senses in word_data.items():
        if sense in senses:
            sense_data[dec] = senses[sense]

    average_change, max_change, peak_decade = compute_weighted_change(
        sense_data,
        max_decade_index=duration
    )

    records.append({
        "Word": word,
        "Sense": sense,
        "Corpus": corpus,
        "duration": duration,
        "event": event,
        "max_weighted_change": max_change,
        "avg_weighted_change": average_change,
        "peak_change_decade": peak_decade
    })

magnitude_df = pd.DataFrame(records)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
TEX_DIR.mkdir(parents=True, exist_ok=True)
output_path = TABLES_DIR / "sense_magnitude_max.csv"
magnitude_df.to_csv(output_path, index=False, encoding="utf-8")

# plot_df = magnitude_df.copy()

# # Only include stabilized senses
# # plot_df = plot_df[plot_df["event"] == 1]

# # Drop missing peaks
# plot_df = plot_df.dropna(subset=["peak_change_decade"])

# # Ensure decade is numeric
# plot_df["peak_decade_start"] = (
#     plot_df["peak_change_decade"]
#     .str.split("-")
#     .str[0]
#     .astype(int)
# )

# counts = (
#     plot_df
#     .groupby(["Corpus", "peak_decade_start"])
#     .size()
#     .reset_index(name="count")
# )

# pivot = counts.pivot(
#     index="peak_decade_start",
#     columns="Corpus",
#     values="count"
# ).fillna(0)

# pivot = pivot.sort_index()

# fig, ax = plt.subplots(figsize=(7, 4))

# pivot.plot(
#     kind="bar",
#     ax=ax,
#     width=0.8
# )

# ax.set_xlabel("Peak Change Decade")
# ax.set_ylabel("Number of Senses")
# ax.set_title("Peak Decade Distribution by Corpus")

# ax.legend(title="Corpus")
# plt.tight_layout()

# tikz_path = "/Users/wenxi/Downloads/diachoronic_model/peak_decade_distribution.tex"
# with open(tikz_path, "w", encoding="utf-8") as f:
#     f.write(r"""\begin{tikzpicture}
# \begin{axis}[
#     ybar,
#     bar width=8pt,
#     width=0.8\textwidth,
#     height=6cm,
#     ylabel={Number of senses},
#     xlabel={Peak change decade},
#     symbolic x coords={""")


#     # x-axis labels
#     decades = list(pivot.index)
#     decade_labels = [f"{d}-{d+9}" for d in decades]
#     f.write(",".join(decade_labels))

#     f.write(r"""},
#     xtick=data,
#     xticklabel style={rotate=45, anchor=east},
#     legend style={
#         at={(0.5,1.05)},
#         anchor=south,
#         legend columns=2,
#         draw=none
#     }
# ]
# """)

#     # One bar series per corpus
#     for corpus in pivot.columns:
#         f.write(r"\addplot coordinates {")

#         for d in decades:
#             label = f"{d}-{d+9}"
#             value = int(pivot.loc[d, corpus])
#             f.write(f"({label},{value}) ")

#         f.write(r"};")
#         f.write("\n")
#         f.write(rf"\addlegendentry{{{corpus}}}")
#         f.write("\n")

#     f.write(r"""
# \end{axis}
# \end{tikzpicture}
# """)

# plt.show()


# -----------------------------
# Load combined data
# -----------------------------
df = pd.read_csv(TABLES_DIR / "sense_magnitude_max.csv")

hongse  = df[df["Corpus"] == "Hongse"]
shenbao = df[df["Corpus"] == "Shenbao"]

# -----------------------------
# TikZ writer
# -----------------------------
def write_tikz_scatter(df, output_path):
    words = sorted(df["Word"].unique())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(r"""\begin{tikzpicture}
\begin{axis}[
    width=0.85\textwidth,
    height=7cm,
    xlabel={Duration of instability (decades)},
    ylabel={Max entropy-weighted magnitude},
    grid=major,
    legend style={
        at={(1.02,1)},
        anchor=north west,
        draw=none,
        font=\small
    },
    scatter/classes={
""")

        markers = [
            "circle", "square", "triangle", "diamond",
            "star", "x", "+", "pentagon",
            "asterisk", "triangle*", "square*", "otimes"
        ]

        for word, marker in zip(words, markers):
            f.write(f"{word}={{mark={marker}}},\n")

        f.write("}\n]\n")

        # Plot each word
        for word in words:
            subset = df[df["Word"] == word]

            f.write(
                r"\addplot[scatter, only marks, scatter src=explicit symbolic] coordinates {"
            )

            for _, row in subset.iterrows():
                x = int(row["duration"])
                y = float(row["avg_weighted_change"])
                f.write(f"({x},{y}) [{word}] ")

            f.write("};\n")
            f.write(f"\\addlegendentry{{{word}}}\n")

        f.write(r"""
\end{axis}
\end{tikzpicture}
""")

# -----------------------------
# Generate TikZ files
# -----------------------------
write_tikz_scatter(hongse, TEX_DIR / "hongse_duration_vs_magnitude.tex")
write_tikz_scatter(shenbao, TEX_DIR / "shenbao_duration_vs_magnitude.tex")


# Ensure average_change is numeric
# ensure numeric
# df["total_weighted_change"] = pd.to_numeric(df["total_weighted_change"])
# df["duration"] = pd.to_numeric(df["duration"])

# # derive average_change
# df["average_change"] = df["total_weighted_change"] / df["duration"]

# # ------------------------------------------------------------------
# # 2. Compute word-level descriptors (by Word × Corpus)
# # ------------------------------------------------------------------
# word_stats = (
#     df
#     .groupby(["Word", "Corpus"])
#     .agg(
#         mean_magnitude=("average_change", "mean"),
#         dispersion=("average_change", "std"),
#         max_magnitude=("average_change", "max"),
#         n_senses=("average_change", "count"),
#     )
#     .reset_index()
# )

# # ------------------------------------------------------------------
# # 3. Dominance ratio
# # ------------------------------------------------------------------
# word_stats["dominance_ratio"] = (
#     word_stats["max_magnitude"] / word_stats["mean_magnitude"]
# )

# # Optional: replace NaN std (single-sense words) with 0
# word_stats["dispersion"] = word_stats["dispersion"].fillna(0.0)

# # ------------------------------------------------------------------
# # 4. Final tidy output
# # ------------------------------------------------------------------
# word_stats = word_stats[
#     ["Word", "Corpus", "n_senses",
#      "mean_magnitude", "dispersion", "dominance_ratio"]
# ]

# print(word_stats)
