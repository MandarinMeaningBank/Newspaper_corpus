from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from transformers import BertModel, BertTokenizerFast


BASE_DIR = Path(__file__).resolve().parent
SENSE_DIR = BASE_DIR / "sense"
RESULT_DIR = BASE_DIR / "result"
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_RANKING_XLSX = BASE_DIR / "副本义项变化程度汇总260704.xlsx"

TARGET_WORDS = [
    "机关",
    "激烈",
    "交通",
    "教授",
    "剧烈",
    "输入",
    "系统",
    "严格",
    "严厉",
    "严重",
    "组织",
    "作业",
]

XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

CORPORA = {
    "shenbao": {
        "sense_dir": SENSE_DIR,
        "pattern": r"processed_(?P<word>.+?)A(?P<sense>\d+)\.txt$",
        "model_dir": BASE_DIR / "bert_chinese_finetune_1930",
    },
    "hongse": {
        "sense_dir": SENSE_DIR / "hongse",
        "pattern": r"(?P<word>.+?)A(?P<sense>\d+)_processed\.txt$",
        "model_dir": BASE_DIR / "bert_hongse_1930",
    },
}


@dataclass
class Example:
    word: str
    sense_id: int
    sentence: str


def iter_examples(corpus_name: str) -> list[Example]:
    config = CORPORA[corpus_name]
    examples: list[Example] = []
    regex = re.compile(config["pattern"])

    for path in sorted(config["sense_dir"].glob("*.txt")):
        match = regex.match(path.name)
        if not match:
            continue

        word = match.group("word")
        if word not in TARGET_WORDS:
            continue

        sense_id = int(match.group("sense"))
        with open(path, encoding="utf-8") as handle:
            for raw_line in handle:
                sentence = raw_line.strip()
                if sentence:
                    examples.append(Example(word=word, sense_id=sense_id, sentence=sentence))

    return examples


def get_word_vector(sentence: str, word: str, model: BertModel, tokenizer: BertTokenizerFast, device: torch.device):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_tokens = tokenizer.tokenize(word)

    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})

    hidden_states = outputs.hidden_states
    last_4_mean = torch.stack(hidden_states[-4:]).mean(dim=0)

    for i in range(len(tokens) - len(word_tokens) + 1):
        if tokens[i : i + len(word_tokens)] == word_tokens:
            idx = list(range(i, i + len(word_tokens)))
            return last_4_mean[0, idx, :].mean(dim=0).cpu()

    return None


def collect_embeddings(corpus_name: str) -> pd.DataFrame:
    config = CORPORA[corpus_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(config["model_dir"])
    model = BertModel.from_pretrained(config["model_dir"], output_hidden_states=True).to(device)
    model.eval()

    records = []
    for example in iter_examples(corpus_name):
        vector = get_word_vector(example.sentence, example.word, model, tokenizer, device)
        if vector is None:
            continue

        records.append(
            {
                "corpus": corpus_name,
                "word": example.word,
                "sense_id": example.sense_id,
                "sentence": example.sentence,
                "embedding": vector.numpy(),
            }
        )

    return pd.DataFrame(records)


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def evaluate_word(df_word: pd.DataFrame) -> tuple[pd.DataFrame, dict] | tuple[None, dict]:
    counts = df_word["sense_id"].value_counts().sort_index()
    min_count = int(counts.min())
    num_senses = int(counts.shape[0])

    if min_count < 2:
        return None, {
            "num_senses": num_senses,
            "num_examples": int(len(df_word)),
            "num_folds": 0,
            "top1": math.nan,
            "top2": math.nan,
            "top2_nontrivial": math.nan,
            "random_top1": 1.0 / num_senses,
            "random_top2": min(2, num_senses) / num_senses,
            "included": False,
            "reason": "min_count_lt_2",
        }

    num_folds = min(5, min_count)
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    X = np.arange(len(df_word))
    y = df_word["sense_id"].to_numpy()
    embeddings = np.stack(df_word["embedding"].to_numpy())

    detailed_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        train_df = df_word.iloc[train_idx]
        test_df = df_word.iloc[test_idx]

        prototypes = {}
        for sense_id, group in train_df.groupby("sense_id"):
            prototypes[sense_id] = np.stack(group["embedding"].to_numpy()).mean(axis=0)

        sense_order = sorted(prototypes.keys())

        for row_idx, row in test_df.iterrows():
            sims = np.array([cosine_similarity_np(row["embedding"], prototypes[sid]) for sid in sense_order])
            probs = softmax(sims)
            ranked = [sense_order[i] for i in np.argsort(-probs)]
            gold = int(row["sense_id"])

            detailed_rows.append(
                {
                    "corpus": row["corpus"],
                    "word": row["word"],
                    "fold": fold_id,
                    "gold_sense": gold,
                    "predicted_sense": ranked[0],
                    "top1_correct": int(ranked[0] == gold),
                    "top2_correct": int(gold in ranked[:2]),
                    "num_senses": num_senses,
                }
            )

    detailed_df = pd.DataFrame(detailed_rows)
    top1 = float(detailed_df["top1_correct"].mean())
    top2 = float(detailed_df["top2_correct"].mean())
    top2_nontrivial = float(detailed_df["top2_correct"].mean()) if num_senses >= 3 else math.nan

    summary = {
        "num_senses": num_senses,
        "num_examples": int(len(df_word)),
        "num_folds": num_folds,
        "top1": top1,
        "top2": top2,
        "top2_nontrivial": top2_nontrivial,
        "random_top1": 1.0 / num_senses,
        "random_top2": min(2, num_senses) / num_senses,
        "included": True,
        "reason": "",
    }
    return detailed_df, summary


def run_instance_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed_frames = []
    summary_rows = []

    for corpus_name in CORPORA:
        cache_path = TABLES_DIR / f"prototype_eval_embeddings_{corpus_name}.pkl"
        if cache_path.exists():
            print(f"Loading cached embeddings for {corpus_name}...")
            df = pd.read_pickle(cache_path)
        else:
            print(f"Collecting embeddings for {corpus_name}...")
            df = collect_embeddings(corpus_name)
            df.to_pickle(cache_path)

        for word in TARGET_WORDS:
            df_word = df[df["word"] == word].reset_index(drop=True)
            if df_word.empty:
                continue

            detailed_df, summary = evaluate_word(df_word)
            summary_rows.append({"corpus": corpus_name, "word": word, **summary})
            if detailed_df is not None:
                detailed_frames.append(detailed_df)

    detailed = pd.concat(detailed_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    detailed.to_csv(TABLES_DIR / "prototype_eval_detailed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES_DIR / "prototype_eval_summary.csv", index=False, encoding="utf-8-sig")
    return detailed, summary


def parse_manual_ranking_sheet() -> pd.DataFrame:
    with ZipFile(MANUAL_RANKING_XLSX) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        try:
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(t.text or "" for t in si.iter(f"{{{XLSX_NS['a']}}}t"))
                for si in sst
            ]
        except KeyError:
            shared_strings = []

        sheet_info = workbook.find("a:sheets", XLSX_NS)[0]
        rid = sheet_info.attrib[f"{{{XLSX_NS['r']}}}id"]
        sheet_xml = ET.fromstring(zf.read("xl/" + rel_map[rid]))

        def col_letters(cell_ref: str) -> str:
            return "".join(ch for ch in cell_ref if ch.isalpha())

        records = []
        for row in sheet_xml.findall(".//a:sheetData/a:row", XLSX_NS)[1:]:
            values = {}
            for cell in row.findall("a:c", XLSX_NS):
                ref = col_letters(cell.attrib["r"])
                t = cell.attrib.get("t")
                value_node = cell.find("a:v", XLSX_NS)
                if value_node is None:
                    value = ""
                else:
                    value = value_node.text or ""
                    if t == "s":
                        value = shared_strings[int(value)]
                values[ref] = value

            if values.get("A"):
                sense_label = values["B"]
                sense_id = int(re.search(r"A(\d+)$", sense_label).group(1)) - 1
                records.append(
                    {
                        "Word": values["A"],
                        "SenseLabel": sense_label,
                        "Sense": sense_id,
                        "human_rank_raw": int(values["D"]),
                    }
                )

    df = pd.DataFrame(records)
    df["human_change_score"] = df.groupby("Word")["human_rank_raw"].transform(lambda s: s.max() + 1 - s)
    return df


def run_manual_shenbao_ranking_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = parse_manual_ranking_sheet()
    model = pd.read_csv(TABLES_DIR / "sense_magnitude_max.csv")
    model = model[model["Corpus"] == "Shenbao"][
        ["Word", "Sense", "max_weighted_change", "avg_weighted_change", "peak_change_decade"]
    ].copy()

    merged = human.merge(model, on=["Word", "Sense"], how="inner")
    merged["model_rank_raw"] = (
        merged.groupby("Word")["max_weighted_change"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    merged["model_change_score"] = merged.groupby("Word")["model_rank_raw"].transform(lambda s: s.max() + 1 - s)
    merged["rank_match"] = (merged["human_rank_raw"] == merged["model_rank_raw"]).astype(int)

    word_rows = []
    for word, group in merged.groupby("Word"):
        corr = spearmanr(group["human_change_score"], group["max_weighted_change"])
        human_top = set(group.loc[group["human_rank_raw"] == group["human_rank_raw"].min(), "Sense"])
        model_top = set(group.loc[group["max_weighted_change"] == group["max_weighted_change"].max(), "Sense"])
        exact_match = int(group["human_rank_raw"].tolist() == group["model_rank_raw"].tolist())

        word_rows.append(
            {
                "Word": word,
                "n_senses": int(group.shape[0]),
                "spearman_rho": float(corr.statistic),
                "p_value": float(corr.pvalue) if corr.pvalue == corr.pvalue else np.nan,
                "top_sense_match": int(bool(human_top & model_top)),
                "exact_rank_match": exact_match,
            }
        )

    word_df = pd.DataFrame(word_rows)
    pooled = spearmanr(merged["human_change_score"], merged["model_change_score"])
    summary_df = pd.DataFrame(
        [
            {
                "metric": "macro_mean_rho",
                "value": float(word_df["spearman_rho"].mean()),
            },
            {
                "metric": "median_rho",
                "value": float(word_df["spearman_rho"].median()),
            },
            {
                "metric": "positive_rho_words",
                "value": float((word_df["spearman_rho"] > 0).sum()),
            },
            {
                "metric": "top_sense_match_words",
                "value": float(word_df["top_sense_match"].sum()),
            },
            {
                "metric": "exact_rank_match_words",
                "value": float(word_df["exact_rank_match"].sum()),
            },
            {
                "metric": "pooled_spearman_rho",
                "value": float(pooled.statistic),
            },
            {
                "metric": "pooled_spearman_p",
                "value": float(pooled.pvalue),
            },
            {
                "metric": "pooled_rank_cell_accuracy",
                "value": float(merged["rank_match"].mean()),
            },
            {
                "metric": "n_words",
                "value": float(word_df.shape[0]),
            },
            {
                "metric": "n_senses_total",
                "value": float(merged.shape[0]),
            },
        ]
    )

    merged.to_csv(TABLES_DIR / "shenbao_manual_ranking_validation_detailed.csv", index=False, encoding="utf-8-sig")
    word_df.to_csv(TABLES_DIR / "shenbao_manual_ranking_validation_by_word.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(TABLES_DIR / "shenbao_manual_ranking_validation_summary.csv", index=False, encoding="utf-8-sig")
    return word_df, summary_df


def print_brief_report(summary: pd.DataFrame) -> None:
    included = summary[summary["included"]].copy()
    corpus_level = (
        included.groupby("corpus", as_index=False)[["top1", "top2"]]
        .mean()
        .rename(columns={"top1": "mean_top1", "top2": "mean_top2"})
    )
    nontrivial = included[included["num_senses"] >= 3].groupby("corpus", as_index=False)["top2"].mean()
    nontrivial = nontrivial.rename(columns={"top2": "mean_top2_nontrivial"})
    corpus_level = corpus_level.merge(nontrivial, on="corpus", how="left")

    overall_top1 = detailed_top1 = included["top1"].mean()
    overall_top2 = included["top2"].mean()

    print("\nInstance-level evaluation")
    print(corpus_level.to_string(index=False))
    print(f"\nOverall mean Top-1: {overall_top1:.4f}")
    print(f"Overall mean Top-2: {overall_top2:.4f}")


if __name__ == "__main__":
    detailed_df, summary_df = run_instance_evaluation()
    manual_word_df, manual_summary_df = run_manual_shenbao_ranking_validation()
    print_brief_report(summary_df)
    print("\nManual Shenbao sense-ranking validation")
    print(manual_word_df.to_string(index=False))
    print(manual_summary_df.to_string(index=False))
