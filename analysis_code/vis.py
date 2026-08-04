import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For custom legend

# --- Configuration ---
word_pinyin_map = {
    "机关": "jiguan", "激烈": "jilie", "交通": "jiaotong", 
    "教授": "jiaoshou", "剧烈": "julie", "输入": "shuru", 
    "系统": "xitong", "严格": "yange", "严厉": "yanli", 
    "严重": "yanzhong", "组织": "zuzhi", "作业": "zuoye"
}
target_words = list(word_pinyin_map.keys())
base_path = "/Users/wenxi/Downloads/diachoronic_model/result"

# Define color mapping for categories
category_colors = {
    "Downgrade": "#e74c3c",  # Red
    "Upgrade": "#2ecc71",    # Green
    "Flat": "#3498db",       # Blue
    "Complex": "#9b59b6"     # Purple
}

# Differentiating words using strictly circles and triangles
# o: circle, ^: triangle_up, v: triangle_down, <: triangle_left, >: triangle_right
distinct_markers = ['o', '>', 'D', 's', 'p', '*']

plt.figure(figsize=(15, 9))

category_marker_usage = {cat: 0 for cat in category_colors}
plot_results = []

# --- 1. Analysis & Regression ---
for word in target_words:
    file_path = os.path.join(base_path, f"new_{word}_shenbao_sense_shift_results.json")
    if not os.path.exists(file_path): continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        word_data = json.load(f)
    
    periods = sorted(word_data.keys())
    entropy_values = [word_data[p]["global_entropy"] for p in periods]
    if len(entropy_values) < 3: continue

    # Linear Regression
    x = np.arange(len(entropy_values))
    y = np.array(entropy_values)
    slope, intercept = np.polyfit(x, y, 1)
    
    # Non-linear detection (MSE)
    y_pred = slope * x + intercept
    mse = np.mean((y - y_pred)**2)

    # Classification
    if mse > 0.015: category = "Complex"
    elif abs(slope) < 0.015: category = "Flat"
    elif slope < -0.015: category = "Downgrade"
    else: category = "Upgrade"

    plot_results.append({
        "pinyin": word_pinyin_map[word],
        "category": category,
        "periods": periods,
        "entropy": entropy_values
    })

# Sort results by category so the legend is organized
plot_results.sort(key=lambda x: x['category'])

# --- 2. Plotting ---
for item in plot_results:
    cat = item['category']
    color = category_colors[cat]
    marker = distinct_markers[category_marker_usage[cat] % len(distinct_markers)]
    category_marker_usage[cat] += 1
    
    # Plotting each word line
    plt.plot(item['periods'], item['entropy'], 
             color=color, 
             marker=marker, 
             markersize=10, 
             linestyle='-', # Solid lines only
             linewidth=2.2, 
             markeredgecolor='white',
             label=f"{item['pinyin']} ({cat})")

# --- 3. Custom Legend and Styling ---
plt.title("Lexical Competition Trajectories Classified by Linear Regression", fontsize=16, fontweight='bold')
plt.ylabel("Global Entropy (Competition Intensity)", fontsize=12)
plt.xlabel("Time Period", fontsize=12)
plt.xticks(rotation=45)

# Standard legend showing each word's line and marker
# We use ncol=1 to list them all clearly
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()