"""Belebele loading, validation and duplicate-passage indexing."""
from __future__ import annotations
import pandas as pd

LANG_CONFIG = {"en": "eng_Latn", "hi": "hin_Deva", "ml": "mal_Mlym"}
REQUIRED_COLUMNS = ["passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4", "correct_answer_num"]

def load_belebele(split: str = "test") -> dict[str, pd.DataFrame]:
    """Download the three official Belebele language configurations."""
    from datasets import load_dataset
    frames = {lang: load_dataset("facebook/belebele", config, split=split).to_pandas()
              for lang, config in LANG_CONFIG.items()}
    return validate_frames(frames)

def validate_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if set(frames) != set(LANG_CONFIG):
        raise ValueError(f"Expected language keys {sorted(LANG_CONFIG)}, got {sorted(frames)}")
    lengths = {lang: len(df) for lang, df in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Language splits have different row counts: {lengths}")
    for lang, df in frames.items():
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{lang} is missing required columns: {missing}")
        frames[lang] = df.reset_index(drop=True).copy()
        frames[lang]["passage"] = frames[lang]["passage"].fillna("").astype(str)
        frames[lang]["question"] = frames[lang]["question"].fillna("").astype(str)
    # Verify stable parallel identifiers if the dataset exposes them.
    candidates = [c for c in ("question_number", "link") if all(c in d.columns for d in frames.values())]
    if candidates:
        col = candidates[0]
        base = frames["en"][col].astype(str).tolist()
        for lang in ("hi", "ml"):
            if frames[lang][col].astype(str).tolist() != base:
                raise ValueError(f"Parallel {col} values do not align for {lang}")
    return frames

def build_passage_corpora(frames: dict[str, pd.DataFrame]):
    """Return unique passages and a row-to-document index for each language."""
    corpora, gold_indices = {}, {}
    for lang, df in frames.items():
        docs, lookup, row_map = [], {}, []
        for passage in df["passage"].tolist():
            if passage not in lookup:
                lookup[passage] = len(docs)
                docs.append(passage)
            row_map.append(lookup[passage])
        corpora[lang] = docs
        gold_indices[lang] = row_map
    # The parallel benchmark should preserve passage-group membership across languages.
    patterns = {lang: [first_occurrence_pattern(x) for x in gold_indices[lang]] for lang in gold_indices}
    if not (patterns["en"] == patterns["hi"] == patterns["ml"]):
        raise ValueError("Passage grouping differs across language configs; inspect parallel alignment before cross-language evaluation.")
    return corpora, gold_indices

def first_occurrence_pattern(indices):
    seen, out = {}, []
    for value in indices:
        if value not in seen:
            seen[value] = len(seen)
        out.append(seen[value])
    return out
