import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from pypinyin import lazy_pinyin

# -----------------------------
# Configuration
# -----------------------------
# target_words = ["机关","激烈","交通","教授","剧烈","输入","严格","严厉","严重","组织","作业"]
target_words = ["机关","输入","严厉"]

base_path = "/Users/wenxi/Downloads/diachoronic_model/result"

def to_pinyin(word):
    return "".join(lazy_pinyin(word))

# -----------------------------
# Entropy function
# -----------------------------
def entropy(p):
    p = np.array(p) + 1e-12
    return -np.sum(p * np.log(p))

# -----------------------------
# Replicator dynamics
# -----------------------------
def simulate(A, x0, steps=200, dt=0.05):

    x = x0.copy()

    for _ in range(steps):

        fitness = A @ x
        avg_fit = x @ fitness

        dx = x * (fitness - avg_fit)
        x = x + dt * dx

        x = x / x.sum()

    return x

# -----------------------------
# Store results
# -----------------------------
rows = []

# -----------------------------
# Main Loop
# -----------------------------
for word in target_words:

    file_path = os.path.join(base_path, f"new_{word}_hongse_sense_shift_results.json")

    if not os.path.exists(file_path):
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        word_data = json.load(f)

    periods = sorted(word_data.keys())

    for period in periods:

        pdata = word_data[period]

        observed_entropy = pdata["global_entropy"]
        x0 = np.array(pdata["global_distribution"])

        counts = np.array(pdata["counts"])

        weights = counts / counts.sum()

        sense_vectors = []

        for k, v in pdata.items():

            if k.isdigit():

                vec = v.get("sense_vector")

                if vec is None:
                    continue

                sense_vectors.append(vec)

                jsd = v.get("JSD")

                if jsd is None or np.isnan(jsd):
                    continue
                sense_id = int(k)

                rows.append({
                    "word": word,
                    "period": period,
                    "sense": f"s{sense_id}",
                    "entropy": observed_entropy,
                    "jsd": jsd,
                    "weight": weights[sense_id]
                })

        # if len(sense_vectors) < 2:
        #     continue

        # sense_vectors = np.array(sense_vectors)

        # # -----------------------------
        # # Cosine similarity matrix
        # # -----------------------------
        # sim_matrix = cosine_similarity(sense_vectors)

        # # -----------------------------
        # # Weighted interaction matrix
        # # -----------------------------
        # A = sim_matrix * weights[np.newaxis, :]

        # # -----------------------------
        # # Run dynamics
        # # -----------------------------
        # x_pred = simulate(A, x0)

        # predicted_entropy = entropy(x_pred)

        # rows.append({
        #     "word": word,
        #     "period": period,
        #     "observed_entropy": observed_entropy,
        #     "predicted_entropy": predicted_entropy
        # })

# -----------------------------
# Convert to dataframe
# -----------------------------
df = pd.DataFrame(rows)

print(df.head())
# -----------------------------
# Correlation test
# -----------------------------
# r, p = pearsonr(df["observed_entropy"], df["predicted_entropy"])

# print("\nCorrelation between observed and predicted entropy:")
# print("r =", r)
# print("p =", p)

# -----------------------------
# Prepare plotting data
# -----------------------------
df["year"] = df["period"].str[:4].astype(int)
df["pinyin"] = df["word"].apply(to_pinyin)
df = df.sort_values(["word","sense","year"])

words = sorted(df["pinyin"].unique())
palette = sns.color_palette("tab10", len(words))

color_map = dict(zip(words, palette))

# -----------------------------
# Plot entropy change
# -----------------------------
# plt.figure(figsize=(10,6))

# for word in df["word"].unique():

#     sub = df[df["word"] == word].sort_values("year")

#     color = color_map[to_pinyin(word)]

#     label = to_pinyin(word)

#     plt.plot(
#         sub["year"],
#         sub["observed_entropy"],
#         marker="o",
#         color=color,
#         linewidth=2,
#         label=f"{label} observed"
#     )

#     plt.plot(
#         sub["year"],
#         sub["predicted_entropy"],
#         linestyle="--",
#         color=color,
#         alpha=0.7,
#         label=f"{label} predicted"
#     )

# plt.xlabel("Year")
# plt.ylabel("Entropy")
# plt.title("Observed vs Predicted Entropy Over Time")
# plt.legend(ncol=2)
# plt.tight_layout()
# plt.show()
# ensure correct ordering of periods
sns.set_style("whitegrid")

words = df["word"].unique()

palette = sns.color_palette("tab10", len(words))
word_colors = dict(zip(words, palette))

sense_markers = {
    "s0": "o",
    "s1": "s",
    "s2": "^",
    "s3": "D"
}

fig, axes = plt.subplots(len(words), 1, figsize=(8, 3*len(words)), sharex=True)

if len(words) == 1:
    axes = [axes]

for ax, word in zip(axes, words):

    wdf = df[df["word"] == word].sort_values("year")

    color = word_colors[word]

    # plot entropy (competition state)
    entropy_series = (
        wdf.groupby("year")["entropy"]
        .first()
        .reset_index()
    )

    ax.plot(
        entropy_series["year"],
        entropy_series["entropy"],
        color="black",
        linewidth=2,
        label="Competition state (entropy)"
    )

    # second axis for JSD
    ax2 = ax.twinx()

    for sense in wdf["sense"].unique():

        sdf = wdf[wdf["sense"] == sense]

        marker = sense_markers.get(sense, "o")

        size = sdf["weight"] * 500

        ax2.scatter(
            sdf["year"],
            sdf["jsd"],
            s=size,
            marker=marker,
            color=color,
            edgecolor="black",
            alpha=0.8,
            label=sense
        )

        ax2.plot(
            sdf["year"],
            sdf["jsd"],
            color=color,
            alpha=0.4
        )

    ax.set_ylabel("Entropy")
    ax2.set_ylabel("JSD")

    ax.set_title(f"Competition–Drift Dynamics: {to_pinyin(word)}")

axes[-1].set_xlabel("Year")

plt.tight_layout()
plt.show()