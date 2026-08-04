import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel
from scipy.stats import entropy
from scipy.spatial.distance import canberra, jensenshannon
import numpy as np
import random
import os
import matplotlib.pyplot as plt
# -----------------------------------------------------
# 1. Configuration
# -----------------------------------------------------
target_word = "组织"
prototype_model_dir = "./bert_hongse_1930"
#prototype_model_dir = "./bert_chinese_finetune_1930"
sense_dir = "./sense/hongse"  # directory containing processed_代价A1.txt ... processed_代价A6.txt
num_senses = 2

# test_files = {
#     "1931-1940": ("data/processed_corpus_2200w1931-1940.txt", "./bert-base-chinese"),
#     "1941-1949": ("data/processed_corpus_2200w1941-1949.txt", "./bert-base-chinese"),
#     "1951-1960": ("data/processed_corpus_2200w1951-1960.txt", "./bert-base-chinese"),
#     "1961-1970": ("data/processed_corpus_2200w1961-1970.txt", "./bert-base-chinese"),
#     "1971-1980": ("data/processed_corpus_2200w1971-1980.txt", "./bert-base-chinese"),
#     "1981-1990": ("data/processed_corpus_2200w1981-1990.txt", "./bert-base-chinese"),
#     "1991-2000": ("data/processed_corpus_2200w1991-2000.txt", "./bert-base-chinese"),
#     "2001-2010": ("data/processed_corpus_2200w2001-2010.txt", "./bert-base-chinese"),
# }

test_files = {
    "1931-1940": ("data/processed_corpus_2200w1931-1940.txt", "./bert_hongse_1930"),
    "1941-1949": ("data/processed_corpus_2200w1941-1949.txt", "./bert_hongse_1940"),
    "1951-1960": ("data/processed_corpus_2200w1951-1960.txt", "./bert_chinese_finetune_1950"),
    "1961-1970": ("data/processed_corpus_2200w1961-1970.txt", "./bert_chinese_finetune_1960"),
    "1971-1980": ("data/processed_corpus_2200w1971-1980.txt", "./bert_chinese_finetune_1970"),
    "1981-1990": ("data/processed_corpus_2200w1981-1990.txt", "./bert_chinese_finetune_1980"),
    "1991-2000": ("data/processed_corpus_2200w1991-2000.txt", "./bert_chinese_finetune_1990"),
    "2001-2010": ("data/processed_corpus_2200w2001-2010.txt", "./bert_chinese_finetune_2000"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------
def get_word_vector(sentence, word, model, tokenizer):
    """Return contextual embedding for one occurrence of target word in sentence."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    hidden_states = outputs.hidden_states
    last_4_mean = torch.stack(hidden_states[-4:]).mean(dim=0)  # [1, seq_len, hidden_size]
    # last_hidden_state = outputs.last_hidden_state 
    # find token indices for word
    word_tokens = tokenizer.tokenize(word)
    for i in range(len(tokens) - len(word_tokens)):
        if tokens[i:i+len(word_tokens)] == word_tokens:
            idx = [j for j in range(i, i+len(word_tokens))]
            return last_4_mean[0, idx, :].mean(dim=0)
    return None


def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_softmax_probs(vec, sense_vecs):
    sims = torch.tensor([cosine_similarity(vec, s) for s in sense_vecs])
    return F.softmax(sims, dim=0).cpu().numpy()


def average_vectors(vectors):
    return torch.stack(vectors).mean(dim=0)

# -----------------------------------------------------
# 3. Step 1: Build prototype vectors from 6 sense files
# -----------------------------------------------------
print("=== Building sense prototype vectors ===")
proto_model = BertModel.from_pretrained(prototype_model_dir, output_hidden_states=True).to(device)
proto_tokenizer = BertTokenizerFast.from_pretrained(prototype_model_dir)

prototype_vectors = []

for sense_id in range(1, num_senses + 1):
    filename = os.path.join(sense_dir, f"{target_word}A{sense_id}_processed.txt")
    if not os.path.exists(filename):
        print(f"⚠️ File not found: {filename}")
        continue

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    selected = random.sample(lines, 100) if len(lines) >= 100 else lines
    sense_vectors = []

    for sent in selected:
        vec = get_word_vector(sent, target_word, proto_model, proto_tokenizer)
        if vec is not None:
            sense_vectors.append(vec)

    if sense_vectors:
        prototype = average_vectors(sense_vectors)
        prototype_vectors.append(prototype)
        print(f"✅ Sense {sense_id}: prototype from {len(sense_vectors)} sentences.")
    else:
        print(f"⚠️ Sense {sense_id}: no valid vectors found.")

if len(prototype_vectors) == 0:
    raise ValueError("No prototypes were successfully built!")

print(f"\n✅ Total prototypes built: {len(prototype_vectors)}")
sense_prototypes = torch.stack(prototype_vectors)

# -----------------------------------------------------
# 4. Step 2: Compute entropy for each test decade
# -----------------------------------------------------
results = {period: {} for period in test_files.keys()}
prev_sense_probs = None
prev_sense_vecs = None

for period, (test_path, model_dir) in test_files.items():
    print(f"\n=== Processing {period} ===")
    model = BertModel.from_pretrained(model_dir, output_hidden_states=True).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)

    with open(test_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if target_word in line]

    # store per-sense lists
    sense_probs = {s: [] for s in range(num_senses)}
    sense_vecs = {s: [] for s in range(num_senses)}
    sense_entropies = {s: [] for s in range(num_senses)}
    sense_assignments = []
    # process all sentences
    for sent in sentences:
        vec = get_word_vector(sent, target_word, model, tokenizer)
        if vec is None:
            continue

        # probability for each sense
        probs = get_softmax_probs(vec, sense_prototypes)
        H = entropy(probs)

        assigned = int(np.argmax(probs))
        sense_assignments.append(assigned)
        sense_probs[assigned].append(probs)
        sense_vecs[assigned].append(vec.cpu().numpy())
        sense_entropies[assigned].append(H)

    # Calculate the Global Distribution (Market Share)
    total_occurrences = len(sense_assignments)
    # Count how many times each sense index (0, 1, 2...) appeared
    counts = np.bincount(sense_assignments, minlength=num_senses)
    global_probs = counts / total_occurrences

    # Calculate Global Entropy for the Decade
    # This H reflects the diversity of the word's usage in the whole corpus
    global_H = entropy(global_probs, base=2)

    # now compute metrics per sense
    results[period] = {
        "global_distribution": global_probs.tolist(),
        "global_entropy": float(global_H),
        "counts": counts.tolist()
    }

    for s in range(num_senses):
        this_period = {}

        # global sense vector
        if sense_probs[s]:
            this_period["sense_vector"] = np.median(sense_probs[s], axis=0).tolist()
        else:
            this_period["sense_vector"] = None

        # entropy
        if sense_entropies[s]:
            this_period["entropy"] = float(np.median(sense_entropies[s]))
        else:
            this_period["entropy"] = None

        # jsd
        if prev_sense_probs is not None and sense_probs[s] and prev_sense_probs[s]:
            prev_avg = np.mean(prev_sense_probs[s], axis=0)
            curr_avg = np.mean(sense_probs[s], axis=0)
            this_period["JSD"] = float(jensenshannon(prev_avg, curr_avg, base=2))
        else:
            this_period["JSD"] = None

        # canberra
        # if prev_sense_vecs is not None and sense_vecs[s] and prev_sense_vecs[s]:
        #     dists = [canberra(x, y) for x in prev_sense_vecs[s] for y in sense_vecs[s]]
        #     this_period["Canberra"] = float(np.mean(dists))
        # else:
        #     this_period["Canberra"] = None

        results[period][s] = this_period

    prev_sense_probs = sense_probs
    prev_sense_vecs = sense_vecs

    print(f"Finished {period}")



out_path = f"new_{target_word}_hongse_sense_shift_results.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results written to {out_path}")
# # -----------------------------------------------------
# # 5. Visualization
# # -----------------------------------------------------
# periods = list(test_files.keys())[1:]  # skip first (no previous-period comparison)

# # ----- ENTROPY -----
# plt.figure(figsize=(10,5))
# for s in range(num_senses):
#     ys = [results[p][s]["entropy"] for p in periods]
#     plt.plot(periods, ys, marker="o", label=f"Sense {s+1}")
# plt.title(f"{target_word} – Entropy per Sense Across Decades")
# plt.xlabel("Decade")
# plt.ylabel("Entropy")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{target_word}_shenbao_entropy_per_sense.pdf")
# plt.show()

# # ----- JSD -----
# plt.figure(figsize=(10,5))
# for s in range(num_senses):
#     ys = [results[p][s]["JSD"] if results[p][s]["JSD"] is not None else 0 for p in periods]
#     plt.plot(periods, ys, marker="s", label=f"Sense {s+1}")
# plt.title(f"{target_word} – Jensen-Shannon Divergence per Sense")
# plt.xlabel("Decade")
# plt.ylabel("JSD")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{target_word}_shenbao_jsd_per_sense.pdf")
# plt.show()

# # ----- CANBERRA -----
# plt.figure(figsize=(10,5))
# for s in range(num_senses):
#     ys = [results[p][s]["Canberra"] if results[p][s]["Canberra"] is not None else 0 for p in periods]
#     plt.plot(periods, ys, marker="^", label=f"Sense {s+1}")
# plt.title(f"{target_word} – Canberra Distance per Sense")
# plt.xlabel("Decade")
# plt.ylabel("Canberra Distance")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{target_word}_shenbao_canberra_per_sense.pdf")
# plt.show()