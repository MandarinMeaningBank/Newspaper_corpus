import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_rel, ttest_ind, spearmanr
from scipy.stats import mannwhitneyu
import os

# --- Configuration ---
target_words = ["机关", "激烈", "交通", "教授", "剧烈", "输入", "系统", "严格", "严厉", "严重", "组织", "作业"]
base_path = "/Users/wenxi/Downloads/diachoronic_model/result"

all_summaries = []
rd_vals = []
rm_vals = []

re_vals = []
ri_vals = []

# def bootstrap_corr(x, y, B=200):
#     vals = []
#     x = np.array(x)
#     y = np.array(y)
#     n = len(x)

#     while len(vals) < B:
#         idx = np.random.choice(n, n, replace=True)

#         xb = x[idx]
#         yb = y[idx]

#         if np.std(xb) == 0 or np.std(yb) == 0:
#             continue

#         r = np.corrcoef(xb, yb)[0,1]

#         if not np.isnan(r):
#             vals.append(r)

#     return vals

# --- Processing Loop ---
for word in target_words:
    file_path = os.path.join(base_path, f"new_{word}_shenbao_sense_shift_results.json")
    
    if not os.path.exists(file_path):
        print(f"Warning: File for '{word}' not found. Skipping.")
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        word_data = json.load(f)
    
    rows = []
    for period, vals in word_data.items():
        if not vals.get("counts") or sum(vals["counts"]) == 0:
            continue
            
        counts = vals["counts"]
        winner_idx = str(np.argmax(counts))
        loser_indices = [str(i) for i in range(len(counts)) if i != np.argmax(counts)]
        
        w_jsd = vals.get(winner_idx, {}).get("JSD")
        l_jsd_values = [vals.get(l_idx, {}).get("JSD") for l_idx in loser_indices]
        valid_l_jsds = [v for v in l_jsd_values if v is not None and not (np.isnan(v) if isinstance(v, float) else False)]
        avg_l_jsd = np.mean(valid_l_jsds) if valid_l_jsds else np.nan
        
        rows.append({
            "Period": period,
            "CS": vals["global_entropy"],
            "Winner_JSD": w_jsd,
            "Loser_JSD": avg_l_jsd
        })

    df_word = pd.DataFrame(rows).dropna().reset_index(drop=True)

    # --- Windowed Logic Containers ---
    window_size = 3

    disruption_decades = []
    entrench_decades = []
    marg_decades = []
    inno_decades = []
    equil_decades = []

    if len(df_word) >= window_size:
        for i in range(len(df_word) - window_size + 1):
            window = df_word.iloc[i : i + window_size]
            # We identify the window by its start and end decade
            window_label = f"{window.iloc[0]['Period'].split('-')[0]}-{window.iloc[-1]['Period'].split('-')[-1]}"

            # Loser Metrics
            r_l, p_l = pearsonr(window["CS"], window["Loser_JSD"])
            # # Winner Metrics
            r_w, p_w = pearsonr(window["CS"], window["Winner_JSD"])
            # boot_l = bootstrap_corr(window["CS"], window["Loser_JSD"], B=200)
            # boot_w = bootstrap_corr(window["CS"], window["Winner_JSD"], B=200)

            if r_l > 0.3 and p_l < 0.1:
                rd_vals.append(abs(r_l))
            if r_l < -0.3 and p_l < 0.1:
                rm_vals.append(abs(r_l))

            if r_w > 0.3 and p_w < 0.1:
                re_vals.append(abs(r_w))
            if r_w < -0.3 and p_w < 0.1:
                ri_vals.append(abs(r_w))
            # for r in boot_l:
            #     if r > 0.3:
            #         rd_vals.append(abs(r))
            #     if r < -0.3:
            #         rm_vals.append(abs(r))

            # for r in boot_w:
            #     if r > 0.3:
            #         re_vals.append(abs(r))
            #     if r < -0.3:
            #         ri_vals.append(abs(r))

            # 1. Disruption check (Loser JSD increases as CS increases)
            r_d, p_d = pearsonr(window["CS"], window["Loser_JSD"])
            if r_d > 0.3 and p_d < 0.1:
                disruption_decades.append(window_label)
            
            # 2. Entrenchment check (Winner JSD decreases as CS decreases)
            r_e, p_e = pearsonr(window["CS"], window["Winner_JSD"])
            if r_e > 0.3 and p_e < 0.1:
                entrench_decades.append(window_label)

            # 3. Marginalization check (Loser JSD decreases as CS increases)
            r_m, p_m = pearsonr(window["CS"], window["Loser_JSD"])
            if r_m < -0.3 and p_m < 0.1:
                marg_decades.append(window_label)

            # 4. Innovation check (Winner JSD increases as CS decreases)
            r_i, p_i = pearsonr(window["CS"], window["Winner_JSD"])
            if r_i < -0.3 and p_i < 0.1:
                inno_decades.append(window_label)
                
            # 5. Equilibrium check (Low volatility in the Competition State)
            if window["CS"].std() < 0.3:
                equil_decades.append(window_label)

    # --- Classification Summary ---
    laws_found = []
    if marg_decades: laws_found.append("Marginalization")
    if inno_decades: laws_found.append("Innovation")
    if disruption_decades: laws_found.append("Disruption")
    if entrench_decades: laws_found.append("Entrenchment")
    
    
    cs_std_total = df_word["CS"].std()
    primary_law = " & ".join(laws_found) if laws_found else ("Equilibrium" if cs_std_total < 0.2 else "Complex")

    all_summaries.append({
        "Word": word,
        "Primary_Law": primary_law,
        "Marg_Decades": ", ".join(marg_decades) if marg_decades else "None",
        "Inno_Decades": ", ".join(inno_decades) if inno_decades else "None",
        "Equil_Decades": ", ".join(equil_decades) if equil_decades else "None",
        "Disrupt_Decades": ", ".join(disruption_decades) if disruption_decades else "None",
        "Entrench_Decades": ", ".join(entrench_decades) if entrench_decades else "None",
        "N_Points": len(df_word)
    })

# --- Final Report ---
summary_df = pd.DataFrame(all_summaries)
print("\n" + "="*60)
print("CORRELATION STRENGTH COMPARISON")
print("="*60)
print(rd_vals)
print(rm_vals)
print("###########")
print(re_vals)
print(ri_vals)
print("###########")
rd_vals = np.array(rd_vals)
rm_vals = np.array(rm_vals)

re_vals = np.array(re_vals)
ri_vals = np.array(ri_vals)


zd_vals = np.arctanh(rd_vals)
zm_vals = np.arctanh(rm_vals)
ze_vals = np.arctanh(re_vals)
zi_vals = np.arctanh(ri_vals)
# paired t-tests
t_dm, p_dm = ttest_ind(zd_vals, zm_vals)
t_ei, p_ei = ttest_ind(ze_vals, zi_vals)

# combine groups
group1 = np.concatenate([zd_vals, ze_vals])
group2 = np.concatenate([zm_vals, zi_vals])

t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
u, p_u = mannwhitneyu(group1, group2, alternative="two-sided")
print("\nStatistical Tests:")
print(f"Disruption vs Marginalization: p = {p_dm:.4f}")
print(f"Entrenchment vs Innovation: p = {p_ei:.4f}")
print(f"Mean Group1 (rd+rm): {np.mean(group1):.4f}")
print(f"Mean Group2 (re+ri): {np.mean(group2):.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.6f}")
print(f"Mann-Whitney U p-value: {p_u:.6f}")

if p_dm < 0.1:
    print(">>> Disruption correlations are significantly stronger than Marginalization")

if p_ei < 0.1:
    print(">>> Entrenchment correlations are significantly stronger than Innovation")
pd.set_option('display.max_colwidth', None) # Ensure full decade strings are visible
print("\n" + "="*110)
print(f"{'WORD':<10} | {'PRIMARY LAW':<25} | {'MARG. PERIODS':<25} | {'INNO. PERIODS':<25} | {'DISRUPT. PERIODS':<25} | {'ENTRENCH. PERIODS':<25}")
print("-" * 110)
for _, row in summary_df.iterrows():
    print(f"{row['Word']:<10} | {row['Primary_Law']:<25} | {row['Marg_Decades']:<25} | {row['Inno_Decades']:<25} | {row['Disrupt_Decades']:<25} | {row['Entrench_Decades']:<25}")
print("="*110)