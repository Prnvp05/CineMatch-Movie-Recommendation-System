"""
Implicit-feedback recommendation with Alternating Least Squares (ALS).

This module builds an implicit user–item matrix from ratings.json,
trains an ALS model (Hu, Koren, Volinsky), persists it, and serves top-N recommendations.

Design goals:
- No compiled dependencies beyond NumPy/SciPy (works on Windows without MSVC)
- ratings.json is the single source of truth
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse

from ml.data_loader import RATINGS_PATH, RatingRow, build_user_item_matrix, load_ratings

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "als_implicit.pkl")


@dataclass
class ALSConfig:
    factors: int = 64
    iterations: int = 15
    reg: float = 0.1
    alpha: float = 40.0  # confidence scaling for implicit feedback
    seed: int = 42


@dataclass
class ALSModelBundle:
    config: ALSConfig
    user_ids: List[int]
    item_ids: List[int]
    user_id_to_index: Dict[int, int]
    item_id_to_index: Dict[int, int]
    X: np.ndarray  # user factors (n_users, factors)
    Y: np.ndarray  # item factors (n_items, factors)
    # Keep the training user-items matrix to support recommend() filtering
    user_items: sparse.csr_matrix
    trained_at: float


def _als_solve(
    fixed_factors: np.ndarray,
    gram: np.ndarray,
    reg: float,
    alpha: float,
    interactions_csr: sparse.csr_matrix,
) -> np.ndarray:
    """
    Solve for one side of implicit ALS (users or items):
      min_x sum_u sum_i c_ui (p_ui - x_u^T y_i)^2 + reg * ||x_u||^2
    where p_ui = 1 if interaction exists, else 0, c_ui = 1 + alpha * r_ui.

    Parameters
    ----------
    fixed_factors : (n_fixed, f) array — the *other* side's current factors
                    (Y when solving for X; X when solving for Y).
    gram          : (f, f) precomputed fixed_factors^T @ fixed_factors.
    reg           : L2 regularisation strength.
    alpha         : confidence scaling factor.
    interactions_csr : (n_entities, n_fixed) sparse matrix — rows are the
                    entities being solved for.
    """
    n_entities = interactions_csr.shape[0]
    n_factors = fixed_factors.shape[1]
    I = np.eye(n_factors, dtype=np.float32)
    out = np.zeros((n_entities, n_factors), dtype=np.float32)

    # base = fixed^T fixed + reg·I  (background term, same for every entity)
    base = gram + (reg * I)

    indptr = interactions_csr.indptr
    indices = interactions_csr.indices
    data = interactions_csr.data

    for u in range(n_entities):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            # No interactions — keep zero vector (cold-start handled in recommend).
            continue

        item_idx = indices[start:end]
        F_i = fixed_factors[item_idx]          # (k, f)
        r_ui = data[start:end].astype(np.float32, copy=False)

        # A = base + F_i^T diag(alpha·r_ui) F_i
        # b = F_i^T (1 + alpha·r_ui)
        conf = alpha * r_ui
        A = base + (F_i.T * conf) @ F_i
        b = (F_i.T * (1.0 + conf)).sum(axis=1)

        out[u] = np.linalg.solve(A, b).astype(np.float32, copy=False)

    return out


def train_als_implicit(
    user_items: sparse.csr_matrix,
    config: ALSConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train implicit ALS on a user-items CSR matrix.
    Returns (X, Y) as float32 arrays — X is user factors, Y is item factors.
    """
    if user_items.shape[0] == 0 or user_items.shape[1] == 0:
        return (
            np.zeros((0, config.factors), dtype=np.float32),
            np.zeros((0, config.factors), dtype=np.float32),
        )

    rng = np.random.default_rng(config.seed)
    n_users, n_items = user_items.shape

    X = rng.normal(0, 0.1, size=(n_users, config.factors)).astype(np.float32)
    Y = rng.normal(0, 0.1, size=(n_items, config.factors)).astype(np.float32)

    item_users = user_items.T.tocsr()  # (n_items, n_users)

    for _ in range(config.iterations):
        # Solve for user factors with item factors fixed
        YtY = (Y.T @ Y).astype(np.float32, copy=False)
        X = _als_solve(Y, YtY, config.reg, config.alpha, user_items)

        # Solve for item factors with user factors fixed
        XtX = (X.T @ X).astype(np.float32, copy=False)
        Y = _als_solve(X, XtX, config.reg, config.alpha, item_users)

    return X, Y


def train_and_save(
    model_path: str = DEFAULT_MODEL_PATH,
    config: Optional[ALSConfig] = None,
    ratings_path: str = RATINGS_PATH,
) -> ALSModelBundle:
    config = config or ALSConfig()
    ratings: List[RatingRow] = load_ratings(ratings_path)
    mat, user_ids, item_ids, u2i, it2i = build_user_item_matrix(ratings)
    X, Y = train_als_implicit(mat, config=config)

    bundle = ALSModelBundle(
        config=config,
        user_ids=user_ids,
        item_ids=item_ids,
        user_id_to_index=u2i,
        item_id_to_index=it2i,
        X=X,
        Y=Y,
        user_items=mat,
        trained_at=time.time(),
    )

    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Optional[ALSModelBundle]:
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, ALSModelBundle):
        return None
    return obj


def recommend_for_user(
    bundle: ALSModelBundle,
    user_id: int,
    n: int = 12,
    exclude_item_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """
    Recommend top-N item IDs for a given user_id from a trained model bundle.

    Returns an empty list for unknown users or cold-start users whose factor
    vector is all zeros (no interactions at training time).
    """
    if user_id not in bundle.user_id_to_index:
        return []

    uidx = bundle.user_id_to_index[int(user_id)]
    user_vec = bundle.X[uidx]

    # FIX: cold-start users have a zero factor vector (no interactions seen
    # during training). Dotting zeros with item vectors yields uniform scores
    # and produces meaningless rankings — return nothing instead.
    if not np.any(user_vec):
        return []

    exclude = set(int(x) for x in (exclude_item_ids or []))

    # Scores: dot each item vector with the user vector
    scores = bundle.Y @ user_vec  # shape (n_items,)

    # Filter already-interacted items
    row = bundle.user_items.getrow(uidx)
    # FIX: use list() to safely index item_ids whether it is a plain list or ndarray
    item_ids_list = list(bundle.item_ids)
    interacted = {item_ids_list[i] for i in row.indices}

    candidates: List[Tuple[int, float]] = []
    for item_index, score in enumerate(scores):
        mid = item_ids_list[item_index]
        if mid in exclude or mid in interacted:
            continue
        candidates.append((mid, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in candidates[:n]]


def recommend_all_users(
    bundle: ALSModelBundle,
    n: int = 12,
) -> Dict[int, List[int]]:
    """
    Generate top-N recommendations for every user in the model.
    """
    out: Dict[int, List[int]] = {}
    for uid in bundle.user_ids:
        out[int(uid)] = recommend_for_user(bundle, uid, n=n)
    return out
