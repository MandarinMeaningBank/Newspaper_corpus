from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Load data
df = pd.read_csv(TABLES_DIR / "regression_data.csv")

# Make sure JSD is numeric
df["JSD"] = pd.to_numeric(df["JSD"], errors="coerce")

# Drop missing JSD values
df = df.dropna(subset=["JSD"])

# We then find the maximum JSD within that group.
df['max_jsd_group'] = df.groupby(['Word', 'Corpus', 'Decade'])['JSD'].transform('max')

# Assign 'dom' to the row where JSD matches the maximum in its group, else 'non'
df['dominance'] = np.where(df['JSD'] == df['max_jsd_group'], 'dom', 'non')

# Drop the temporary calculation column
df = df.drop(columns=['max_jsd_group'])

# Keep only noun and verb
# df = df[df["POS"].isin(["noun", "verb"])].copy()
epsilon = 1e-8  # prevents log(0)
df = df[df["JSD"] > 0].copy()
df["log_JSD"] = np.log(df["JSD"] + epsilon)

# Convert categorical variables
df["Word"] = df["Word"].astype("category")
df["Sense"] = df["Sense"].astype("category")
df["Corpus"] = df["Corpus"].astype("category")
df["POS"] = df["POS"].astype("category")
df["dominance"] = df["dominance"].astype("category")
# Set hongse as reference category (optional but recommended)
df["Corpus"] = df["Corpus"].cat.reorder_categories(["Hongse", "Shenbao"])
# Set noun as reference category (optional but recommended)
df["POS"] = df["POS"].cat.reorder_categories(["verb", "noun", "adj"])
df["dominance"] = df["dominance"].cat.reorder_categories(["non", "dom"])
# Ensure freq is numeric
df["Freq"] = pd.to_numeric(df["Freq"], errors="coerce")

# Remove missing values
df = df.dropna(subset=["Freq"])

# Log-transform frequency (standard in corpus linguistics)
df["log_freq"] = np.log(df["Freq"] + 1)
# Optional: center it for mixed models
df["log_freq_c"] = df["log_freq"] - df["log_freq"].mean()

# Convert decade to numeric midpoint (e.g., 1941-1949 → 1941)
df["Decade_start"] = df["Decade"].str.split("-").str[0].astype(int)

# Center decade (recommended for mixed models)
df["Decade_c"] = df["Decade_start"] - df["Decade_start"].mean()

# --- compute polysemy per Word × Corpus ---
polysemy = df.groupby(["Word","Corpus"])["Sense"].nunique().rename("polysemy")

df = df.merge(polysemy, on=["Word","Corpus"])

# log transform
df["log_polysemy"] = np.log(df["polysemy"])

# center
df["log_polysemy_c"] = df["log_polysemy"] - df["log_polysemy"].mean()

print(df.columns)
print("###########")

model = smf.mixedlm(
    "JSD ~ Decade_c + log_freq_c + log_polysemy_c + C(Corpus) + C(POS)",   # fixed effects
    df,
    groups=df["Word"],                 # random intercept for Word
    vc_formula={"Sense": "0 + C(Sense)"}  # random intercept for Sense
)



result = model.fit()

print(result.summary())

#####this is the regression for freq and jsd
# fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# corpora = ["Hongse", "Shenbao"]

# for i, corp in enumerate(corpora):
    
#     subset = df[df["Corpus"] == corp]
    
#     sns.scatterplot(
#         data=subset,
#         x="log_freq",
#         y="JSD",
#         hue="POS",
#         palette="Set1",
#         alpha=0.6,
#         ax=axes[i]
#     )
    
#     # regression line
#     sns.regplot(
#         data=subset,
#         x="log_freq",
#         y="JSD",
#         scatter=False,
#         color="black",
#         ax=axes[i]
#     )
    
#     axes[i].set_title(f"Corpus: {corp}")
#     axes[i].set_xlabel("Log Frequency")
#     axes[i].set_ylabel("Semantic Drift (JSD)")
#     axes[i].grid(alpha=0.3)

# plt.tight_layout()
# plt.show()

#####this is the regression for POS and decade

fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)

corpora = ["Hongse", "Shenbao"]

for i, corp in enumerate(corpora):

    subset = df[df["Corpus"] == corp]

    # scatter points (POS color)
    sns.scatterplot(
        data=subset,
        x="Decade_start",
        y="JSD",
        hue="POS",
        palette="Set1",
        alpha=0.6,
        ax=axes[i]
    )

    # regression lines by POS
    sns.regplot(
        data=subset[subset["POS"]=="noun"],
        x="Decade_start",
        y="JSD",
        scatter=False,
        ax=axes[i],
        color="red",
        label="noun"
    )

    sns.regplot(
        data=subset[subset["POS"]=="verb"],
        x="Decade_start",
        y="JSD",
        scatter=False,
        ax=axes[i],
        color="blue",
        label="verb"
    )

    sns.regplot(
        data=subset[subset["POS"]=="adj"],
        x="Decade_start",
        y="JSD",
        scatter=False,
        ax=axes[i],
        color="green",
        label="adj"
    )

    axes[i].set_title(f"Corpus: {corp}")
    axes[i].set_xlabel("Decade")
    axes[i].set_ylabel("Semantic Drift (JSD)")
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# -------- CREATE EFFECT GRID (fixed effects only) --------
# -------- STEP 1: Create prediction grid --------
# 1. Define the unique categories we want to visualize

# print("--- Coordinates for TikZ ---")
# for i, corp in enumerate(corpora):
#     panel_num = i + 1
#     print(f"\n% --- Paste inside \\nextgroupplot #{panel_num} (Corpus: {corp}) ---")
#     print("% Place these commands BEFORE the ribbon/line plots so they are in the background.")
    
#     for pos in pos_types:
#         # 1. Filter data for this specific combination
#         subset = df[(df['Corpus'] == corp) & (df['POS'] == pos)]
        
#         # 2. Sample points (handle cases where data might be scarce)
#         n_samples = min(len(subset), SAMPLES_PER_GROUP)
        
#         if n_samples > 0:
#             # Use random_state for reproducible plots
#             sampled_data = subset.sample(n=n_samples, random_state=42)
            
#             # 3. Format coordinates into a TikZ-readable string: (x1,y1) (x2,y2) ...
#             coords_str = " ".join([
#                 f"({x:.2f},{y:.5f})" 
#                 for x, y in zip(sampled_data['Decade_c'], sampled_data['JSD'])
#             ])
            
#             # 4. Determine the color name based on your LaTeX definitions
#             color_name = f"pos{pos}" # e.g., yields 'posnoun', 'posverb'
            
#             # 5. Print the final LaTeX command
#             print(f"% Background points for {pos.capitalize()}")
#             print(f"\\addplot[only marks, mark=*, mark size=0.8pt, color={color_name}, opacity=0.25] coordinates {{{coords_str}}};")
#         else:
#             print(f"% No data available for {pos} in {corp}")

# print("\n% ==========================================")
# print("% === END COPY-PASTE BLOCK ===============")
# print("% ==========================================")
# for corp in corpora:
#     ax = axes[corp]
#     # Filter raw data for the background of this specific subplot
#     sns.scatterplot(data=df[df['Corpus']==corp], x='Decade_c', y='JSD', 
#                     hue='POS', palette=colors, alpha=0.2, ax=ax, legend=False)
#     for pos in pos_types:
#         temp_predict = pd.DataFrame({
#             'Decade_c': decade_range,
#             'Corpus': corp,
#             'POS': pos
#         })
        
#         temp_predict['fit'] = result.predict(temp_predict)
        
#         # Calculate Uncertainty (CI)
#         design_matrix = dmatrix("Decade_c + Corpus", temp_predict, return_type='dataframe')
#         fe_names = design_matrix.columns
#         cov_matrix_aligned = result.cov_params().loc[fe_names, fe_names]
#         pred_var = np.diag(design_matrix @ cov_matrix_aligned @ design_matrix.T)
        
#         lower = temp_predict['fit'] - 1.96 * np.sqrt(pred_var)
#         upper = temp_predict['fit'] + 1.96 * np.sqrt(pred_var)
        
#         # 3. Plot the line
#         ax.plot(decade_range, temp_predict['fit'], color=colors[pos], lw=3, label=pos)
#         ax.fill_between(decade_range, lower, upper, color=colors[pos], alpha=0.15)
    
#     ax.set_title(f"Corpus: {corp}")
#     ax.set_xlabel("Decade (Centered)")

# ax1.set_ylabel("Semantic Drift (JSD)")
# plt.legend(title="POS")
# plt.tight_layout()
# plt.show()
