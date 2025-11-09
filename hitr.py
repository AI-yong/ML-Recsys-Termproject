# ============================================
# inference_recommender.py
# Cross-domain Hybrid Recommendation (SBERT + PCA + GMM)
# ============================================

import os, math, random, sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import joblib

# -----------------------------
# 0. 경로 및 전역 설정
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 폴더 자동 인식

TARGET_CSV = os.path.join(BASE_DIR, "df_goodreads_final.csv")
TGT_EMB_CACHE = os.path.join(BASE_DIR, "tgt_embs_f16.npy")
TGT_IDS_CACHE = os.path.join(BASE_DIR, "tgt_isbn.npy")

PCA_PATH   = os.path.join(BASE_DIR, "pca_model.joblib")
GMM_PATH   = os.path.join(BASE_DIR, "gmm_model.joblib")
META_PATH  = os.path.join(BASE_DIR, "meta_info.pkl")
SBERT_PATH = os.path.join(BASE_DIR, "sbert_model")

# -----------------------------
# 기본 하이퍼파라미터
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MIN_POS_RATING = 4
MIN_USER_INTERACTIONS = 5
K_DEFAULT = 10
TITLE_WEIGHT = 2
AUTHOR_WEIGHT = 1
USE_REVIEW = True
MAX_REVIEWS_PER_ISBN_TGT = 2
DEFAULT_ALPHA = 0.40
DEFAULT_GAMMA = 1.20
DIVERSIFY_SIM_TH = 0.80
PRESELECT_M = 250
AUTHOR_BONUS = 0.10
TITLE_BONUS = 0.04
POP_BONUS = 0.03

# -----------------------------
# 1. 모델 로드
# -----------------------------
def load_models():
    global ALPHA, GAMMA, pca, gmm, model
    meta = joblib.load(META_PATH)
    ALPHA = meta.get("ALPHA", DEFAULT_ALPHA)
    GAMMA = meta.get("GAMMA", DEFAULT_GAMMA)
    print(f"[META] ALPHA={ALPHA}, GAMMA={GAMMA}")

    pca = joblib.load(PCA_PATH)
    gmm = joblib.load(GMM_PATH)
    print("[Loaded] PCA, GMM")

    # SBERT 로드
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    model = SentenceTransformer(SBERT_PATH, device=device)
    print(f"[Loaded] SBERT ({device})")

# -----------------------------
# 2. 데이터 로드
# -----------------------------
def load_df(path):
    df = pd.read_csv(path)
    needed = {"User-ID","ISBN","Book-Rating","Book-Title","Book-Author","Review"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df["User-ID"] = df["User-ID"].astype(str)
    df["ISBN"] = df["ISBN"].astype(str)
    df["Book-Rating"] = pd.to_numeric(df["Book-Rating"], errors="coerce").fillna(0).clip(0,5).astype(int)
    for c in ["Book-Title","Book-Author","Review"]:
        df[c] = df[c].fillna("")
    return df

def build_item_docs(df):
    if USE_REVIEW:
        try:
            sampled = df.groupby("ISBN", group_keys=False).apply(
                lambda g: g.sample(n=min(MAX_REVIEWS_PER_ISBN_TGT, len(g)), random_state=RANDOM_SEED)
            )
        except Exception:
            sampled = df.sort_values("ISBN").groupby("ISBN", group_keys=False).head(MAX_REVIEWS_PER_ISBN_TGT)
        reviews = sampled.groupby("ISBN")["Review"].apply(lambda s: " ".join(s.tolist()))
        reviews = pd.DataFrame({"ISBN": reviews.index, "reviews": reviews.values}).set_index("ISBN")
    else:
        reviews = pd.DataFrame(index=df["ISBN"].unique()); reviews["reviews"] = ""

    titles  = df.drop_duplicates("ISBN")[["ISBN","Book-Title"]].set_index("ISBN")
    authors = df.drop_duplicates("ISBN")[["ISBN","Book-Author"]].set_index("ISBN")
    agg = titles.copy()
    agg["Book-Author"] = authors["Book-Author"]
    agg["reviews"] = reviews["reviews"].reindex(agg.index).fillna("")
    agg["text"] = (
        ((agg["Book-Title"] + " ") * TITLE_WEIGHT) +
        ((agg["Book-Author"] + " ") * AUTHOR_WEIGHT) +
        (agg["reviews"] if USE_REVIEW else "")
    ).str.strip()
    return agg.reset_index()[["ISBN","text"]]

# -----------------------------
# 3. SBERT 임베딩 (캐시)
# -----------------------------
def encode_texts(texts, batch_size=512):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        E = model.encode(batch, batch_size=len(batch), show_progress_bar=False, normalize_embeddings=True)
        embs.append(E.astype(np.float32))
    return np.vstack(embs).astype(np.float32)

def embed_with_cache(items_df):
    if os.path.exists(TGT_EMB_CACHE) and os.path.exists(TGT_IDS_CACHE):
        X = np.load(TGT_EMB_CACHE)
        ids = np.load(TGT_IDS_CACHE, allow_pickle=True).tolist()
        print("[Cache] Loaded embeddings.")
        return X, ids
    texts = items_df["text"].tolist()
    X = encode_texts(texts)
    X = normalize(X).astype(np.float16)
    ids = items_df["ISBN"].tolist()
    np.save(TGT_EMB_CACHE, X)
    np.save(TGT_IDS_CACHE, np.array(ids, dtype=object))
    print("[Cache] Saved embeddings.")
    return X, ids

# -----------------------------
# 4. PCA + GMM + 유저 프로필
# -----------------------------
def compute_topic_probs(X_tgt32):
    X_pca = pca.transform(X_tgt32)
    P = gmm.predict_proba(X_pca).astype(np.float32)
    P = np.power(P, GAMMA)
    P /= (P.sum(axis=1, keepdims=True) + 1e-8)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return P

def rating_weight(r):
    return {5:1.5, 4:1.0, 3:0.4, 2:0.2, 1:0.1}.get(r, 0.0)

def build_user_profiles(df, item_index, P_items, X_tgt32):
    user_prob, user_emb, user_hist = {}, {}, {}
    rng = np.random.default_rng(RANDOM_SEED)
    for uid, g in tqdm(df.groupby("User-ID"), desc="User profiles"):
        if len(g) < MIN_USER_INTERACTIONS: continue
        g = g.sample(frac=1.0, random_state=int(rng.integers(0, 1e9)))
        pairs = list(zip(g["ISBN"], g["Book-Rating"]))
        user_hist[uid] = pairs
        train_pairs = pairs[:-1]
        # topic
        num=None; denom=0
        for bid,r in train_pairs:
            idx=item_index.get(bid); w=rating_weight(r)
            if idx is None or w<=0: continue
            num = P_items[idx]*w if num is None else num+P_items[idx]*w
            denom+=w
        if num is not None and denom>0:
            u=num/(denom+1e-8); u/=np.linalg.norm(u)+1e-8
            user_prob[uid]=u.astype(np.float32)
        # embed
        num=None; denom=0
        for bid,r in train_pairs:
            idx=item_index.get(bid); w=rating_weight(r)
            if idx is None or w<=0: continue
            num = X_tgt32[idx]*w if num is None else num+X_tgt32[idx]*w
            denom+=w
        if num is not None and denom>0:
            ue=num/(denom+1e-8); ue/=np.linalg.norm(ue)+1e-8
            user_emb[uid]=ue.astype(np.float32)
    return user_prob, user_emb, user_hist

# -----------------------------
# 5. 추천 함수
# -----------------------------
def score_candidates_hybrid(u_prob, u_emb, cands, item_index, P_items, X_tgt32):
    idxs=[item_index[c] for c in cands if c in item_index]
    s_prob = P_items[idxs] @ u_prob
    s_emb  = X_tgt32[idxs] @ u_emb
    return ALPHA*s_prob + (1-ALPHA)*s_emb

def recommend_for_user(
    uid: str,
    K: int,
    ALL_ITEMS: list,
    item_index: dict,
    user_prob: dict,
    user_emb: dict,
    P_items: np.ndarray,
    X_tgt32: np.ndarray,
    user_hist: dict,
):
    if uid not in user_prob or uid not in user_emb:
        print(f"[WARN] User {uid} not found.")
        return []

    seen = set(i for i, _ in user_hist[uid])
    cand_pool = [i for i in ALL_ITEMS if i not in seen]

    if not cand_pool:
        return []

    scores = score_candidates_hybrid(user_prob[uid], user_emb[uid],
                                     cand_pool, item_index, P_items, X_tgt32)

    k = min(K, len(scores))
    top_idx = np.argpartition(-scores, k-1)[:k]
    top_sorted = top_idx[np.argsort(-scores[top_idx])]
    ranked = [(cand_pool[i], float(scores[i])) for i in top_sorted]

    return ranked


# -----------------------------
# 6. main()
# -----------------------------
def main():
    print("=== Cross-domain Inference Recommender ===")
    load_models()
    df = load_df(TARGET_CSV)
    print(f"[Data] {df.shape}")
    items = build_item_docs(df)
    X_tgt, ids = embed_with_cache(items)
    X_tgt32 = X_tgt.astype(np.float32)
    item_index = {bid:i for i,bid in enumerate(ids)}
    P_items = compute_topic_probs(X_tgt32)
    user_prob, user_emb, user_hist = build_user_profiles(df, item_index, P_items, X_tgt32)
    print(f"[Users] valid={len(user_prob)}")

    if user_prob:
        uid = next(iter(user_prob.keys()))
        print(f"\n[Example] User: {uid}")
        recs = recommend_for_user(uid, 10, ids, item_index, user_prob, user_emb, P_items, X_tgt32, user_hist)
        for r, (bid,score) in enumerate(recs,1):
            title = df.loc[df["ISBN"]==bid,"Book-Title"].iloc[0]
            print(f"{r:2d}. {title} (score={score:.3f})")
    else:
        print("No valid users found.")

if __name__ == "__main__":
    main()
