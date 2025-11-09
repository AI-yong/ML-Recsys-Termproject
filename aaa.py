# -*- coding: utf-8 -*-
"""
Cross-domain LDA Content-based Recommender (Lightweight, fits your columns)
- SOURCE : df_amazon_final.csv
- TARGET : df_goodreads_final.csv
- Columns expected (exact as your screenshot):
  ['User-ID','ISBN','Book-Rating','Book-Title','Book-Author','Publisher','Review']

Pipeline
1) Aggregate texts per ISBN (title*2 + author + sampled reviews)
2) Build shared dictionary on (SOURCE_items + TARGET_items) texts
3) Train LDA on SOURCE_items only
4) Infer topic vectors for TARGET_items
5) Build user profiles on TARGET (leave-one-out) using rating-weighted mean
6) Evaluate with sampled negatives: P@K, R@K, F1@K, nDCG@K, MAP@K
"""

import os
import gc
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import (
    preprocess_string, strip_punctuation, strip_non_alphanum, strip_numeric,
    strip_multiple_whitespaces, strip_short
)

# ==============================
# Config (tune for speed/quality)
# ==============================
SOURCE_CSV = "df_amazon_final.csv"
TARGET_CSV = "df_goodreads_final.csv"

RANDOM_SEED = 42
MIN_POS_RATING = 4           # positive label threshold (1..5)
MIN_USER_INTERACTIONS = 5    # users with >=5 interactions kept

# Text aggregation
MAX_REVIEWS_PER_ISBN = 3     # take up to N reviews per ISBN (speed/memory)
TITLE_WEIGHT = 2             # replicate title to weight it higher
AUTHOR_WEIGHT = 1

# Dictionary/Corpus slimming
MAX_VOCAB = 40000            # keep top-N tokens by doc freq
MIN_DOC_FREQ = 5             # token must appear in at least N docs
NO_ABOVE = 0.5               # drop tokens that appear in >50% of docs

# LDA (make it light)
NUM_TOPICS = 40
LDA_PASSES = 2
LDA_ITER = 50
LDA_WORKERS = max(1, (os.cpu_count() or 2) - 1)
CHUNK_SIZE = 2000

# Evaluation
K_LIST = [10, 20]
NEGATIVE_SAMPLES = 500       # sampled negatives per user
USE_ALL_ITEMS = False        # True = All-Items evaluation (slow)

# ==============================
# Utilities
# ==============================
CUSTOM_FILTERS = [
    lambda x: x.lower(),
    strip_punctuation,
    strip_non_alphanum,
    strip_numeric,
    strip_multiple_whitespaces,
    lambda x: strip_short(x, minsize=2),
]

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = ""
    return preprocess_string(text, CUSTOM_FILTERS)

def safe_concat(parts: List[str]) -> str:
    return " ".join([p for p in parts if isinstance(p, str) and p.strip()])

# ==============================
# Load & basic checks
# ==============================
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"User-ID","ISBN","Book-Rating","Book-Title","Book-Author","Review"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    # Types
    df["User-ID"] = df["User-ID"].astype(str)
    df["ISBN"] = df["ISBN"].astype(str)
    # Rating to numeric clipped to [1,5]
    if not np.issubdtype(df["Book-Rating"].dtype, np.number):
        df["Book-Rating"] = pd.to_numeric(df["Book-Rating"], errors="coerce")
    df["Book-Rating"] = df["Book-Rating"].fillna(0).clip(lower=0, upper=5).astype(int)
    # Fill text NA
    for c in ["Book-Title","Book-Author","Review"]:
        df[c] = df[c].fillna("")
    return df

# ==============================
# Aggregate texts per ISBN
# ==============================
def build_item_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame: columns = ['ISBN','agg_text']
    - agg_text = (title*TITLE_WEIGHT + author*AUTHOR_WEIGHT + up to N reviews)
    """
    # Sample up to N reviews per ISBN (fast & balanced)
    # groupby.sample is pandas>=1.1
    try:
        sampled = df.groupby("ISBN", group_keys=False).apply(
            lambda g: g.sample(n=min(MAX_REVIEWS_PER_ISBN, len(g)), random_state=RANDOM_SEED)
        )
    except Exception:
        # Fallback: take head(N)
        sampled = df.sort_values("ISBN").groupby("ISBN", group_keys=False).head(MAX_REVIEWS_PER_ISBN)

    # Pick first title/author seen for that ISBN
    titles = df.drop_duplicates("ISBN")[["ISBN","Book-Title"]].set_index("ISBN")
    authors = df.drop_duplicates("ISBN")[["ISBN","Book-Author"]].set_index("ISBN")

    reviews = sampled.groupby("ISBN")["Review"].apply(lambda s: " ".join(s.astype(str).tolist()))

    agg = pd.DataFrame({"ISBN": reviews.index, "reviews_join": reviews.values})
    agg = agg.set_index("ISBN")
    agg["Book-Title"] = titles["Book-Title"]
    agg["Book-Author"] = authors["Book-Author"]
    agg = agg.fillna("")
    # Weighted concat
    agg["agg_text"] = (
        ((agg["Book-Title"] + " ") * TITLE_WEIGHT) +
        ((agg["Book-Author"] + " ") * AUTHOR_WEIGHT) +
        agg["reviews_join"]
    ).str.strip()
    return agg.reset_index()[["ISBN","agg_text"]]

def texts_to_tokens(df_items: pd.DataFrame) -> List[List[str]]:
    return [tokenize(t) for t in df_items["agg_text"].tolist()]

# ==============================
# Dictionary & corpora
# ==============================
def make_dictionary(all_texts: List[List[str]]) -> Dictionary:
    dct = Dictionary(all_texts)
    dct.filter_extremes(no_below=MIN_DOC_FREQ, no_above=NO_ABOVE, keep_n=MAX_VOCAB)
    dct.compactify()
    return dct

def to_bow(texts: List[List[str]], dct: Dictionary):
    return [dct.doc2bow(t) for t in texts]

# ==============================
# LDA helpers
# ==============================
def train_lda(corpus, dct: Dictionary) -> LdaMulticore:
    model = LdaMulticore(
        corpus=corpus,
        id2word=dct,
        num_topics=NUM_TOPICS,
        passes=LDA_PASSES,
        iterations=LDA_ITER,
        chunksize=CHUNK_SIZE,
        workers=LDA_WORKERS,
        random_state=RANDOM_SEED,
        eval_every=None,
    )
    return model

def dense_topic(model: LdaMulticore, bow) -> np.ndarray:
    vec = np.zeros(NUM_TOPICS, dtype=np.float32)
    for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
        vec[tid] = prob
    # L2 normalize for cosine via dot
    n = np.linalg.norm(vec) + 1e-8
    return (vec / n).astype(np.float32)

# ==============================
# User profiling (TARGET)
# ==============================
def build_user_profiles_target(
    df_target: pd.DataFrame,
    item_vecs: Dict[str, np.ndarray],
    min_interactions: int = MIN_USER_INTERACTIONS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[str, int]]]]:
    """
    - Leave-one-out per user (random shuffle because no timestamp)
    - Returns:
      user_profile: {user -> topic vec}
      user_hist: {user -> [(isbn, rating)]} shuffled list
    """
    user_profile = {}
    user_hist = {}
    rng = np.random.default_rng(RANDOM_SEED)

    for uid, g in df_target.groupby("User-ID"):
        if len(g) < min_interactions:
            continue
        # shuffle rows
        g = g.sample(frac=1.0, random_state=int(rng.integers(0, 1e9)))
        pairs = list(zip(g["ISBN"].astype(str).tolist(), g["Book-Rating"].astype(int).tolist()))
        user_hist[uid] = pairs

        # Train part = all except last one (test)
        train_pairs = pairs[:-1] if len(pairs) >= 2 else pairs
        sum_vec = None
        sum_w = 0.0
        for isbn, r in train_pairs:
            v = item_vecs.get(isbn)
            if v is None:
                continue
            # weight: normalized rating (0..1)
            w = max(0.0, min(1.0, (float(r) - 1.0) / 4.0))
            sum_vec = (v * w) if sum_vec is None else (sum_vec + v * w)
            sum_w += w
        if sum_vec is None or sum_w == 0.0:
            # skip users with no usable items
            user_hist.pop(uid)  # drop hist too
            continue
        prof = sum_vec / (sum_w + 1e-8)
        prof = (prof / (np.linalg.norm(prof) + 1e-8)).astype(np.float32)
        user_profile[uid] = prof

    return user_profile, user_hist

# ==============================
# Candidate generation & ranking
# ==============================
def sampled_candidates(uid: str, train_items: set, pos_items: set, all_items: List[str]) -> List[str]:
    # Always include positives
    pool = [i for i in pos_items if i not in train_items]
    # Negatives from unseen set
    ban = train_items.union(pos_items)
    neg_pool = [i for i in all_items if i not in ban]
    n = min(NEGATIVE_SAMPLES, len(neg_pool))
    if n > 0:
        rng = np.random.default_rng(RANDOM_SEED + hash(uid) % 100000)
        pool.extend(rng.choice(neg_pool, size=n, replace=False).tolist())
    return pool

def all_items_candidates(train_items: set, all_items: List[str]) -> List[str]:
    return [i for i in all_items if i not in train_items]

def rank_scores(user_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # cosine since vectors are L2-normalized → dot product
    return mat @ user_vec

# ==============================
# Metrics
# ==============================
def precision_at_k(rels: List[int], k: int) -> float:
    return float(sum(rels[:k])) / float(k)

def recall_at_k(rels: List[int], k: int, num_pos: int) -> float:
    if num_pos == 0:
        return 0.0
    return float(sum(rels[:k])) / float(num_pos)

def f1_at_k(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else (2 * p * r) / (p + r)

def dcg_at_k(rels: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        dcg += (2**rel - 1.0) / math.log2(i + 1.0)
    return dcg

def ndcg_at_k(rels: List[int], k: int) -> float:
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return 0.0 if idcg == 0 else dcg_at_k(rels, k) / idcg

def ap_at_k(rels: List[int], k: int) -> float:
    hits, ap = 0, 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel == 1:
            hits += 1
            ap += hits / i
    return 0.0 if hits == 0 else ap / hits

# ==============================
# Main
# ==============================
def main():
    seed_everything()

    print("1) Load CSVs")
    src = load_df(SOURCE_CSV)       # Amazon
    tgt = load_df(TARGET_CSV)       # Goodreads

    print("2) Aggregate item texts (SOURCE & TARGET)")
    src_items = build_item_texts(src)    # ['ISBN','agg_text']
    tgt_items = build_item_texts(tgt)

    # Build shared dictionary on concatenated texts (no label leakage; only text)
    print("3) Tokenize & build shared dictionary")
    src_tokens = texts_to_tokens(src_items)
    tgt_tokens = texts_to_tokens(tgt_items)
    dct = make_dictionary(src_tokens + tgt_tokens)
    print(f"  - vocab size: {len(dct)}")

    print("4) Build BoW corpora (items)")
    src_corpus = [dct.doc2bow(t) for t in src_tokens]
    tgt_corpus = [dct.doc2bow(t) for t in tgt_tokens]

    print("5) Train LDA on SOURCE items only (light)")
    lda = train_lda(src_corpus, dct)

    # Infer topic vectors for TARGET items
    print("6) Infer topic vectors for TARGET items")
    tgt_item_ids = tgt_items["ISBN"].tolist()
    tgt_item_vecs = {}
    tgt_mat = []
    for isbn, bow in zip(tgt_item_ids, tgt_corpus):
        v = dense_topic(lda, bow)
        tgt_item_vecs[isbn] = v
        tgt_mat.append(v)
    # unique ordering
    uniq_item_ids = list(dict.fromkeys(tgt_item_ids))
    item_index = {i: idx for idx, i in enumerate(uniq_item_ids)}
    item_mat = np.vstack([tgt_item_vecs[i] for i in uniq_item_ids]).astype(np.float32)

    # Free memory
    del src_tokens, tgt_tokens, src_corpus, tgt_corpus
    gc.collect()

    print("7) Build user profiles on TARGET (leave-one-out)")
    user_profile, user_hist = build_user_profiles_target(tgt, tgt_item_vecs, MIN_USER_INTERACTIONS)
    print(f"  - valid users: {len(user_profile)}")

    print("8) Evaluate (SampledNegatives={} / AllItems={})".format(NEGATIVE_SAMPLES, USE_ALL_ITEMS))
    metrics = {f"P@{k}": [] for k in K_LIST}
    metrics.update({f"R@{k}": [] for k in K_LIST})
    metrics.update({f"F1@{k}": [] for k in K_LIST})
    metrics.update({f"nDCG@{k}": [] for k in K_LIST})
    metrics.update({f"MAP@{k}": [] for k in K_LIST})

    all_items = uniq_item_ids

    for uid, prof in user_profile.items():
        pairs = user_hist[uid]
        if len(pairs) < 2:
            continue
        # LOO: last one as test
        train_pairs = pairs[:-1]
        test_item, test_rating = pairs[-1]
        if int(test_rating) < MIN_POS_RATING:
            # only evaluate when held-out is positive
            continue
        gold = {test_item}
        train_set = set(i for i, _ in train_pairs)

        # candidates
        if USE_ALL_ITEMS:
            cands = all_items_candidates(train_set, all_items)
        else:
            cands = sampled_candidates(uid, train_set, gold, all_items)
        if not cands:
            continue

        cand_mat = np.vstack([tgt_item_vecs[i] for i in cands]).astype(np.float32)
        scores = cand_mat @ prof  # cosine since normalized
        order = np.argsort(-scores)
        ranked = [cands[i] for i in order]
        rels = [1 if i in gold else 0 for i in ranked]

        for k in K_LIST:
            p = precision_at_k(rels, k)
            r = recall_at_k(rels, k, num_pos=len(gold))
            f1 = f1_at_k(p, r)
            nd = ndcg_at_k(rels, k)
            ap = ap_at_k(rels, k)
            metrics[f"P@{k}"].append(p)
            metrics[f"R@{k}"].append(r)
            metrics[f"F1@{k}"].append(f1)
            metrics[f"nDCG@{k}"].append(nd)
            metrics[f"MAP@{k}"].append(ap)

    print("\n=== RESULTS (mean over users) ===")
    for k in K_LIST:
        P = float(np.mean(metrics[f"P@{k}"])) if metrics[f"P@{k}"] else 0.0
        R = float(np.mean(metrics[f"R@{k}"])) if metrics[f"R@{k}"] else 0.0
        F1 = float(np.mean(metrics[f"F1@{k}"])) if metrics[f"F1@{k}"] else 0.0
        ND = float(np.mean(metrics[f"nDCG@{k}"])) if metrics[f"nDCG@{k}"] else 0.0
        MAP = float(np.mean(metrics[f"MAP@{k}"])) if metrics[f"MAP@{k}"] else 0.0
        print(f"K={k:>2} | P@{k}: {P:.4f} | R@{k}: {R:.4f} | F1@{k}: {F1:.4f} | nDCG@{k}: {ND:.4f} | MAP@{k}: {MAP:.4f}")

if __name__ == "__main__":
    main()
