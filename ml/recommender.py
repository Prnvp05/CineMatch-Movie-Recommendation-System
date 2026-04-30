"""
Movie Recommendation System
- Content-Based Filtering: TF-IDF on genres, cast, director
- Collaborative Filtering: SVD Matrix Factorization
- Hybrid: Weighted blend of both
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import logging
import threading

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

from ml.data_loader import load_movies, load_ratings


class ContentBasedRecommender:
    """Recommends movies based on genre, cast, and director similarity using TF-IDF + Cosine Similarity."""

    def __init__(self):
        self.movies = None
        self.tfidf_matrix = None
        self.movie_indices = None
        self.vectorizer = None

    def _build_feature_string(self, movie):
        genres_val = movie.get('genres', [])
        if isinstance(genres_val, list):
            # FIX: explicitly skip None and non-string-convertible values to
            # prevent str(None) == "None" leaking into TF-IDF as a real token.
            genres = ' '.join([
                str(g).strip()
                for g in genres_val
                if g is not None and str(g).strip()
            ])
        else:
            genres = str(genres_val).replace('|', ' ') if genres_val is not None else ''

        cast_raw = movie.get('cast') or []
        cast = ' '.join([
            a.replace(' ', '_')
            for a in cast_raw
            if a is not None
        ])

        director_raw = movie.get('director') or ''
        director = director_raw.replace(' ', '_')

        # Weight director and genres more heavily
        return f"{genres} {genres} {cast} {director} {director} {director}"

    def fit(self, movies):
        self.movies = movies
        self.movie_indices = {int(m['id']): i for i, m in enumerate(movies)}

        features = [self._build_feature_string(m) for m in movies]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(features)
        logger.info(f"Content-based model fitted on {len(movies)} movies")

    def recommend(self, movie_ids, n=10, exclude_ids=None):
        if not movie_ids:
            return []
        exclude_ids = set(exclude_ids or [])
        valid_ids = [int(mid) for mid in movie_ids if int(mid) in self.movie_indices]
        if not valid_ids:
            return []

        indices = [self.movie_indices[mid] for mid in valid_ids]
        user_profile = np.asarray(self.tfidf_matrix[indices].mean(axis=0))
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix)[0]

        scored = [
            (i, sim_scores[i])
            for i in range(len(self.movies))
            if int(self.movies[i]['id']) not in exclude_ids
            and int(self.movies[i]['id']) not in valid_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [int(self.movies[i]['id']) for i, _ in scored[:n]]

    def recommend_by_preferences(self, genres, directors, actors, n=10, exclude_ids=None):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        exclude_ids = set(exclude_ids or [])
        genre_str = ' '.join(genres)
        director_str = ' '.join([d.replace(' ', '_') for d in directors])
        actor_str = ' '.join([a.replace(' ', '_') for a in actors])
        pref_text = f"{genre_str} {genre_str} {actor_str} {director_str} {director_str} {director_str}"

        pref_vec = self.vectorizer.transform([pref_text])
        sim_scores = cosine_similarity(pref_vec, self.tfidf_matrix)[0]
        scored = [
            (i, sim_scores[i])
            for i in range(len(self.movies))
            if int(self.movies[i]['id']) not in exclude_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [int(self.movies[i]['id']) for i, _ in scored[:n]]


class CollaborativeFilteringRecommender:
    """SVD-based Matrix Factorization for collaborative filtering."""

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}

    def fit(self, ratings_data):
        df = pd.DataFrame(ratings_data)
        df['rating'] = df['rating'].astype(float)
        df['userId'] = df['userId'].astype(int)
        df['movieId'] = df['movieId'].astype(int)

        self.global_mean = df['rating'].mean()
        users = df['userId'].unique()
        items = df['movieId'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: i for i, it in enumerate(items)}
        self.reverse_item_map = {i: it for it, i in self.item_map.items()}

        n_users = len(users)
        n_items = len(items)

        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        for epoch in range(self.n_epochs):
            shuffled = df.sample(frac=1, random_state=epoch)
            total_loss = 0

            # FIX: replaced iterrows() with itertuples() — significantly faster
            # for large datasets since it avoids constructing a Series per row.
            for row in shuffled.itertuples(index=False):
                u = self.user_map.get(row.userId)
                i = self.item_map.get(row.movieId)
                if u is None or i is None:
                    continue

                pred = (
                    self.global_mean
                    + self.user_biases[u]
                    + self.item_biases[i]
                    + self.user_factors[u] @ self.item_factors[i]
                )
                err = row.rating - pred
                total_loss += err ** 2

                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])
                uf = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * uf)
                self.item_factors[i] += self.lr * (err * uf - self.reg * self.item_factors[i])

            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(total_loss / len(df))
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        logger.info("Collaborative filtering model trained")

    def recommend_for_user(self, user_id, liked_movie_ids=None, n=10, exclude_ids=None):
        exclude_ids = set(exclude_ids or [])
        u = self.user_map.get(int(user_id)) if user_id is not None else None

        # If the user isn't in the training set, infer a pseudo-user embedding
        # from liked movies to still enable CF-style ranking.
        if u is None:
            liked_movie_ids = [
                int(mid) for mid in (liked_movie_ids or [])
                if int(mid) in self.item_map
            ]
            if not liked_movie_ids:
                return []
            liked_idx = [self.item_map[mid] for mid in liked_movie_ids]
            pseudo_user = np.mean(self.item_factors[liked_idx], axis=0)
            pseudo_bias = float(np.mean(self.item_biases[liked_idx]))
            scored = []
            for movie_id, i in self.item_map.items():
                if int(movie_id) in exclude_ids:
                    continue
                score = (
                    self.global_mean
                    + pseudo_bias
                    + self.item_biases[i]
                    + pseudo_user @ self.item_factors[i]
                )
                scored.append((int(movie_id), float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [mid for mid, _ in scored[:n]]

        scored = []
        for movie_id, i in self.item_map.items():
            if int(movie_id) in exclude_ids:
                continue
            score = (
                self.global_mean
                + self.user_biases[u]
                + self.item_biases[i]
                + self.user_factors[u] @ self.item_factors[i]
            )
            scored.append((int(movie_id), float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in scored[:n]]


class HybridRecommender:
    """Combines content-based and collaborative filtering."""

    def __init__(self, content_weight=0.4, collab_weight=0.6):
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.content_model = ContentBasedRecommender()
        self.collab_model = CollaborativeFilteringRecommender()
        self.movies = None
        self.movie_map = {}
        self.is_trained = False

    def fit(self, movies, ratings):
        self.movies = movies
        self.movie_map = {int(m['id']): m for m in movies}
        self.content_model.fit(movies)
        self.collab_model.fit(ratings)
        self.is_trained = True
        self._save()
        logger.info("Hybrid recommender trained and saved")

    def _save(self):
        path = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'content': self.content_model,
                'collab': self.collab_model,
                'movies': self.movies,
                'movie_map': self.movie_map,
            }, f)

    @classmethod
    def load(cls):
        path = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            data = pickle.load(f)
        rec = cls()
        rec.content_model = data['content']
        rec.collab_model = data['collab']
        rec.movies = data['movies']
        rec.movie_map = data['movie_map']
        rec.is_trained = True
        return rec

    def get_movie(self, movie_id):
        if movie_id is None:
            return None
        return self.movie_map.get(int(movie_id))

    @staticmethod
    def _movie_id(movie):
        mid = movie.get("id", None)
        if mid is None:
            mid = movie.get("movieId", None)
        if mid is None:
            return None
        try:
            return int(mid)
        except Exception:
            return None

    def _movies_from_ids(self, ids):
        out = []
        for mid in ids:
            m = self.movie_map.get(int(mid))
            if m:
                out.append(m)
        return out

    def search_movies(self, query, n=30):
        q = (query or "").strip().lower()
        if not q:
            return []
        hits = []
        for m in self.movies or []:
            title = str(m.get("title", "")).lower()
            if q in title:
                hits.append(m)
        hits.sort(key=lambda m: (m.get("popularity", 0), m.get("rating", 0)), reverse=True)
        return hits[:n]

    def get_popular_movies(self, n=20, exclude_ids=None):
        exclude_ids = set(int(x) for x in (exclude_ids or []))
        movies = []
        for m in (self.movies or []):
            mid = self._movie_id(m)
            if mid is None or mid in exclude_ids:
                continue
            movies.append(m)
        movies.sort(key=lambda m: (m.get("popularity", 0), m.get("rating", 0)), reverse=True)
        return movies[:n]

    def get_all_genres(self):
        genres = set()
        for m in self.movies or []:
            for g in (m.get("genres") or []):
                if g:
                    genres.add(str(g))
        return sorted(genres)

    def get_all_directors(self):
        directors = set()
        for m in self.movies or []:
            d = m.get("director")
            if d:
                directors.add(str(d))
        return sorted(directors)

    def get_all_actors(self):
        actors = set()
        for m in self.movies or []:
            for a in (m.get("cast") or []):
                if a:
                    actors.add(str(a))
        return sorted(actors)

    def recommend_new_user(self, liked_movie_ids, genres, directors, actors, exclude_ids=None, n=12):
        exclude_ids = set(exclude_ids or [])
        liked_movie_ids = [int(mid) for mid in (liked_movie_ids or [])]
        if liked_movie_ids:
            ids = self.content_model.recommend(
                liked_movie_ids, n=max(n, 30), exclude_ids=exclude_ids
            )
        else:
            ids = self.content_model.recommend_by_preferences(
                genres or [], directors or [], actors or [],
                n=max(n, 30), exclude_ids=exclude_ids,
            )
        return self._movies_from_ids(ids)[:n]

    def recommend(self, user_id, liked_movie_ids, exclude_ids=None, n=12):
        exclude_ids = set(exclude_ids or [])
        liked_movie_ids = [int(mid) for mid in (liked_movie_ids or [])]

        # Content-based recommendations
        content_recs = self.content_model.recommend(
            liked_movie_ids, n=30, exclude_ids=exclude_ids
        )
        # FIX: guard against ZeroDivisionError when content_recs is empty
        # (e.g. a new user with no liked movies that are known to the model).
        if content_recs:
            n_content = len(content_recs)
            content_scores = {
                mid: (n_content - i) / n_content
                for i, mid in enumerate(content_recs)
            }
        else:
            content_scores = {}

        # Collaborative recommendations
        collab_recs = self.collab_model.recommend_for_user(
            user_id, liked_movie_ids=liked_movie_ids, n=30, exclude_ids=exclude_ids
        )
        # FIX: same guard for collab side.
        if collab_recs:
            n_collab = len(collab_recs)
            collab_scores = {
                mid: (n_collab - i) / n_collab
                for i, mid in enumerate(collab_recs)
            }
        else:
            collab_scores = {}

        # If both sides are empty there is nothing to recommend.
        all_candidates = set(content_scores.keys()) | set(collab_scores.keys())
        if not all_candidates:
            return []

        # Combine scores
        final_scores = {}
        for mid in all_candidates:
            cs = content_scores.get(mid, 0)
            co = collab_scores.get(mid, 0)
            final_scores[mid] = self.content_weight * cs + self.collab_weight * co

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for mid, score in ranked[:n]:
            if int(mid) in self.movie_map:
                movie = dict(self.movie_map[int(mid)])
                movie['rec_score'] = round(score, 4)
                result.append(movie)
        return result


# ---------------------------------------------------------------------------
# Global singleton — protected by a lock so concurrent requests in a threaded
# server (e.g. Flask with threaded=True, gunicorn) don't double-train.
# ---------------------------------------------------------------------------
_recommender: Optional["HybridRecommender"] = None  # noqa: F821
_recommender_lock = threading.Lock()


def get_recommender() -> "HybridRecommender":
    global _recommender
    # FIX: double-checked locking prevents two threads from both seeing None
    # and racing to train a new model simultaneously.
    if _recommender is None:
        with _recommender_lock:
            if _recommender is None:
                _recommender = HybridRecommender.load()
                if _recommender is None:
                    logger.info("Training new recommender model...")
                    movies = [m for m in load_movies() if m.get("id") is not None]
                    ratings = [
                        {"userId": r.user_id, "movieId": r.movie_id, "rating": r.rating}
                        for r in load_ratings()
                    ]
                    _recommender = HybridRecommender()
                    _recommender.fit(movies, ratings)
    return _recommender


def train_and_save() -> "HybridRecommender":
    global _recommender
    movies = [m for m in load_movies() if m.get("id") is not None]
    ratings = [
        {"userId": r.user_id, "movieId": r.movie_id, "rating": r.rating}
        for r in load_ratings()
    ]
    new_model = HybridRecommender()
    new_model.fit(movies, ratings)
    # FIX: update the global atomically so in-flight requests on the old model
    # are not interrupted mid-use.
    with _recommender_lock:
        _recommender = new_model
    return _recommender


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rec = train_and_save()
    print("Model trained!")
