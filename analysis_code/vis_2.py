import json
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations

# -----------------------------
# Configuration
# -----------------------------
word = "交通"   # change to any word
window_size = 3

base_path = "/Users/wenxi/Downloads/diachoronic_model/result"
file_path = os.path.join(base_path, f"new_{word}_hongse_sense_shift_results.json")

# -----------------------------
# Load data
# -----------------------------
with open(file_path, "r", encoding="utf-8") as f:
    word_data = json.load(f)

rows = []

for period, vals in word_data.items():

    row = {"Period": period}

    for k, v in vals.items():
        if k.isdigit():
            row[f"sense_{k}"] = v["JSD"]

    rows.append(row)

df = pd.DataFrame(rows).sort_values("Period").reset_index(drop=True)

sense_cols = [c for c in df.columns if c.startswith("sense_")]

# -----------------------------
# Detect cooperative windows
# -----------------------------
coop_windows = []

pairs = list(combinations(sense_cols, 2))

for i in range(len(df) - window_size + 1):

    window = df.iloc[i:i+window_size]

    for s1, s2 in pairs:

        sub = window[[s1, s2]].dropna()

        if len(sub) < window_size:
            continue

        r, p = pearsonr(sub[s1], sub[s2])

        if r > 0.3 and p < 0.1:

            coop_windows.append((
                window.iloc[0]["Period"],
                window.iloc[-1]["Period"]
            ))

# -----------------------------
# Prepare plotting data
# -----------------------------
df_long = df.melt(
    id_vars="Period",
    value_vars=sense_cols,
    var_name="Sense",
    value_name="JSD"
)

# df_long["Period_idx"] = range(len(df_long) // len(sense_cols)) * len(sense_cols)

# -----------------------------
# Plot style
# -----------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,4))

palette = sns.color_palette("colorblind", len(sense_cols))

# -----------------------------
# Plot sense trajectories
# -----------------------------
sns.lineplot(
    data=df_long,
    x="Period",
    y="JSD",
    hue="Sense",
    marker="o",
    linewidth=2,
    palette=palette
)

# -----------------------------
# Highlight cooperation windows
# -----------------------------
for start, end in coop_windows:

    start_idx = df[df["Period"] == start].index[0]
    end_idx = df[df["Period"] == end].index[0]

    plt.axvspan(
        start_idx - 0.2,
        end_idx + 0.2,
        color="gray",
        alpha=0.2
    )

# -----------------------------
# Labels and formatting
# -----------------------------
plt.title(f"Semantic Trajectories of Senses for '{word}'", fontsize=13)
plt.ylabel("Semantic Drift (JSD)")
plt.xlabel("Time Period")

plt.xticks(rotation=45)

plt.legend(
    title="Sense",
    frameon=False
)

sns.despine()

plt.tight_layout()

# -----------------------------
# Save publication-quality figure
# -----------------------------
plt.savefig(
    f"{word}_sense_trajectories.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()