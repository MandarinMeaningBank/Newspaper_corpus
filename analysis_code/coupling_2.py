import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics.pairwise import cosine_similarity
import os
from itertools import combinations

# -----------------------------
# Configuration
# -----------------------------
target_words = ["机关","激烈","交通","教授","剧烈","输入","系统","严格","严厉","严重","组织","作业"]

base_path = "/Users/wenxi/Downloads/diachoronic_model/result"

window_size = 3

results = []

coop_sims = []
alt_sims = []

# -----------------------------
# Main Loop
# -----------------------------
for word in target_words:

    file_path = os.path.join(base_path, f"new_{word}_shenbao_sense_shift_results.json")

    if not os.path.exists(file_path):
        continue

    with open(file_path,"r",encoding="utf-8") as f:
        word_data = json.load(f)

    rows = []
    vectors = {}

    # -----------------------------
    # Extract data
    # -----------------------------
    for period,vals in word_data.items():

        if not vals.get("counts"):
            continue

        row = {"Period":period}
        vectors[period] = {}

        for sid,svals in vals.items():

            if sid.isdigit():

                sname = f"sense_{sid}"
                row[sname] = svals.get("JSD")

                if "sense_vector" in svals:
                    vectors[period][sname] = np.array(svals["sense_vector"])

        rows.append(row)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)

    sense_cols = [c for c in df.columns if c.startswith("sense_")]

    if len(df) < window_size:
        continue

    sense_pairs = list(combinations(sense_cols,2))

    # -----------------------------
    # Sliding windows
    # -----------------------------
    for i in range(len(df)-window_size+1):

        window = df.iloc[i:i+window_size]

        start = window.iloc[0]["Period"].split("-")[0]
        end = window.iloc[-1]["Period"].split("-")[-1]

        window_label = f"{start}-{end}"

        for s1,s2 in sense_pairs:

            sub = window[[s1,s2]].dropna()

            if len(sub) < window_size:
                continue

            r,p = pearsonr(sub[s1],sub[s2])

            if r > 0.3 and p < 0.1:

                # -----------------------------
                # PERIOD-LEVEL SIMILARITIES
                # -----------------------------
                sim12 = []

                for period in window["Period"]:

                    if s1 in vectors[period] and s2 in vectors[period]:

                        v1 = vectors[period][s1].reshape(1,-1)
                        v2 = vectors[period][s2].reshape(1,-1)

                        sim12.append(cosine_similarity(v1,v2)[0][0])

                if len(sim12) == 0:
                    continue

                coop_sim = np.mean(sim12)

                # -----------------------------
                # ALTERNATIVE COMPARISONS
                # -----------------------------
                for s3 in sense_cols:

                    if s3 in [s1,s2]:
                        continue

                    sim13 = []
                    sim23 = []

                    for period in window["Period"]:

                        if s1 in vectors[period] and s3 in vectors[period]:

                            v1 = vectors[period][s1].reshape(1,-1)
                            v3 = vectors[period][s3].reshape(1,-1)

                            sim13.append(
                                cosine_similarity(v1,v3)[0][0]
                            )

                        if s2 in vectors[period] and s3 in vectors[period]:

                            v2 = vectors[period][s2].reshape(1,-1)
                            v3 = vectors[period][s3].reshape(1,-1)

                            sim23.append(
                                cosine_similarity(v2,v3)[0][0]
                            )

                    # -----------------------------
                    # v1-v2 vs v1-v3
                    # -----------------------------
                    if len(sim13) > 0:

                        alt13 = np.mean(sim13)

                        coop_sims.append(coop_sim)
                        alt_sims.append(alt13)

                        results.append({
                            "Word":word,
                            "Window":window_label,
                            "Coop_Pair":f"{s1}-{s2}",
                            "Alt_Pair":f"{s1}-{s3}",
                            "Coop_Sim":coop_sim,
                            "Alt_Sim":alt13
                        })

                    # -----------------------------
                    # v1-v2 vs v2-v3
                    # -----------------------------
                    if len(sim23) > 0:

                        alt23 = np.mean(sim23)

                        coop_sims.append(coop_sim)
                        alt_sims.append(alt23)

                        results.append({
                            "Word":word,
                            "Window":window_label,
                            "Coop_Pair":f"{s1}-{s2}",
                            "Alt_Pair":f"{s2}-{s3}",
                            "Coop_Sim":coop_sim,
                            "Alt_Sim":alt23
                        })

# -----------------------------
# Results
# -----------------------------
df_results = pd.DataFrame(results)

print("\n"+"="*100)
print("COOPERATION VS ALTERNATIVE SIMILARITY")
print("="*100)
print(df_results)

# -----------------------------
# Paired statistical test
# -----------------------------
if len(coop_sims) > 1:

    t,p = ttest_rel(coop_sims,alt_sims)

    print("\n"+"="*60)
    print("PAIRED SIMILARITY TEST")
    print("="*60)

    print("Mean cooperative similarity:",np.mean(coop_sims))
    print("Mean alternative similarity:",np.mean(alt_sims))
    print("t statistic:",t)
    print("p value:",p)