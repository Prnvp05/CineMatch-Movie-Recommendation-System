"""
Single source of truth for recommendation data: data/ratings.json (and movies.json for metadata).

This module centralizes:
- loading/saving ratings.json with schema normalization
- upserting/deleting user–movie ratings (used by like/watch/rate endpoints)
- building sparse user-item matrices for ALS and other models
- computing popularity from the same ratings data (fallback)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.json")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.json")


@dataclass(frozen=True)
class RatingRow:
    user_id: int
    movie_id: int
    rating: float


def _atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_movies(path: str = MOVIES_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    # Normalize schema: ensure id/movieId are ints and present
    for m in movies:
        if "id" not in m and "movieId" in m:
            m["id"] = m["movieId"]
        if "movieId" not in m and "id" in m:
            m["movieId"] = m["id"]
        try:
            if m.get("id") is not None:
                m["id"] = int(m["id"])
        except Exception:
            m["id"] = None
        try:
            if m.get("movieId") is not None:
                m["movieId"] = int(m["movieId"])
        except Exception:
            m["movieId"] = m.get("id")
    return movies


def load_ratings(path: str = RATINGS_PATH) -> List[RatingRow]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f) or []
    out: List[RatingRow] = []
    for r in raw:
        try:
            uid = int(r.get("userId", r.get("user_id")))
            mid = int(r.get("movieId", r.get("movie_id")))
            rating = float(r.get("rating", 0))
        except Exception:
            continue
        if mid is None or uid is None:
            continue
        if rating <= 0:
            continue
        out.append(RatingRow(user_id=uid, movie_id=mid, rating=rating))
    return out


def save_ratings(rows: Sequence[RatingRow], path: str = RATINGS_PATH) -> None:
    payload = [{"userId": int(r.user_id), "movieId": int(r.movie_id), "rating": float(r.rating)} for r in rows]
    _atomic_write_json(path, payload)


def upsert_rating(user_id: int, movie_id: int, rating: float, path: str = RATINGS_PATH) -> None:
    """
    Insert or update a single (user_id, movie_id) rating in ratings.json.
    """
    rows = load_ratings(path)
    uid = int(user_id)
    mid = int(movie_id)
    rt = float(rating)
    found = False
    new_rows: List[RatingRow] = []
    for r in rows:
        if r.user_id == uid and r.movie_id == mid:
            new_rows.append(RatingRow(uid, mid, rt))
            found = True
        else:
            new_rows.append(r)
    if not found:
        new_rows.append(RatingRow(uid, mid, rt))
    save_ratings(new_rows, path=path)


def delete_rating(user_id: int, movie_id: int, path: str = RATINGS_PATH) -> None:
    rows = load_ratings(path)
    uid = int(user_id)
    mid = int(movie_id)
    new_rows = [r for r in rows if not (r.user_id == uid and r.movie_id == mid)]
    save_ratings(new_rows, path=path)


def get_user_history(user_id: int, path: str = RATINGS_PATH) -> Dict[int, float]:
    """
    Returns {movie_id: rating} for a given user.
    """
    uid = int(user_id)
    hist: Dict[int, float] = {}
    for r in load_ratings(path):
        if r.user_id == uid:
            hist[int(r.movie_id)] = float(r.rating)
    return hist


def build_user_item_matrix(
    rows: Sequence[RatingRow],
) -> Tuple[sparse.csr_matrix, List[int], List[int], Dict[int, int], Dict[int, int]]:
    """
    Build a sparse user-item matrix from ratings.json rows.
    Values are the rating values (float32).
    """
    if not rows:
        empty = sparse.csr_matrix((0, 0), dtype=np.float32)
        return empty, [], [], {}, {}

    user_ids = sorted({int(r.user_id) for r in rows})
    item_ids = sorted({int(r.movie_id) for r in rows})
    u2i = {u: idx for idx, u in enumerate(user_ids)}
    i2i = {m: idx for idx, m in enumerate(item_ids)}

    rr = np.fromiter((u2i[int(r.user_id)] for r in rows), dtype=np.int32)
    cc = np.fromiter((i2i[int(r.movie_id)] for r in rows), dtype=np.int32)
    dd = np.fromiter((float(r.rating) for r in rows), dtype=np.float32)
    mat = sparse.csr_matrix((dd, (rr, cc)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    mat.sum_duplicates()
    return mat, user_ids, item_ids, u2i, i2i


def compute_popular_movie_ids(
    rows: Sequence[RatingRow],
    n: int = 20,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """
    Popularity fallback from ratings.json:
    rank by (count desc, avg_rating desc).
    """
    exclude = set(int(x) for x in (exclude_ids or []))
    if not rows:
        return []

    stats: Dict[int, Tuple[int, float]] = {}  # movie_id -> (count, sum)
    for r in rows:
        mid = int(r.movie_id)
        if mid in exclude:
            continue
        c, s = stats.get(mid, (0, 0.0))
        stats[mid] = (c + 1, s + float(r.rating))

    ranked = []
    for mid, (c, s) in stats.items():
        avg = s / max(c, 1)
        ranked.append((mid, c, avg))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [mid for mid, _c, _a in ranked[:n]]

