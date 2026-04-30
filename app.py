"""
CineMatch - Movie Recommendation System
Flask Backend with SQLite database
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
import sys
import logging
import random
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))
from ml.recommender import get_recommender
from ml.als_implicit import (
    ALSModelBundle,
    load_model as _load_als_model_from_disk,
    recommend_for_user as als_recommend_for_user,
    train_and_save as train_als_and_save,
)
from ml.data_loader import (
    load_ratings,
    upsert_rating,
    delete_rating,
    get_user_history,
    compute_popular_movie_ids,
    RATING_MIN,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# FIX: secret key must come from the environment so it is not baked into source
# code. Fall back to a random value for local dev (sessions won't survive
# restarts, which is acceptable in development).
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cinematch.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# FIX: harden session cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Enable SESSION_COOKIE_SECURE = True in production (requires HTTPS).
# app.config['SESSION_COOKIE_SECURE'] = True

db = SQLAlchemy(app)

# ratings.json user id namespace for app users (avoid clashing with existing dataset userIds)
APP_USER_ID_OFFSET = 10_000_000

# ---------------------------------------------------------------------------
# FIX: cache the ALS bundle in memory so it is not unpickled from disk on
# every single recommendation request.
# ---------------------------------------------------------------------------
_als_bundle: ALSModelBundle | None = None
_als_bundle_lock = threading.Lock()


def get_als_bundle() -> ALSModelBundle | None:
    """Return the cached ALS model bundle, loading from disk at most once."""
    global _als_bundle
    if _als_bundle is None:
        with _als_bundle_lock:
            if _als_bundle is None:
                _als_bundle = _load_als_model_from_disk()
    return _als_bundle


def refresh_als_bundle() -> ALSModelBundle | None:
    """Train a fresh ALS model, persist it, and update the in-memory cache."""
    global _als_bundle
    bundle = train_als_and_save()
    with _als_bundle_lock:
        _als_bundle = bundle
    return bundle


# ─────────────────────────── MODELS ───────────────────────────

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    liked_movies = db.Column(db.Text, default='[]')       # JSON list of movie IDs
    watched_movies = db.Column(db.Text, default='[]')     # JSON list of movie IDs
    fav_genres = db.Column(db.Text, default='[]')
    fav_directors = db.Column(db.Text, default='[]')
    fav_actors = db.Column(db.Text, default='[]')
    onboarding_done = db.Column(db.Boolean, default=False)
    ratings = db.Column(db.Text, default='{}')            # {movieId: rating}

    def get_liked(self):
        # FIX: always return List[int] so comparisons with int movie IDs are safe
        return [int(x) for x in json.loads(self.liked_movies or '[]')]

    def get_watched(self):
        # FIX: always return List[int]
        return [int(x) for x in json.loads(self.watched_movies or '[]')]

    def get_genres(self):
        return json.loads(self.fav_genres or '[]')

    def get_directors(self):
        return json.loads(self.fav_directors or '[]')

    def get_actors(self):
        return json.loads(self.fav_actors or '[]')

    def get_ratings(self):
        return json.loads(self.ratings or '{}')

    def set_liked(self, lst):
        # FIX: always persist as List[int] to keep the stored type consistent
        self.liked_movies = json.dumps([int(x) for x in lst])

    def set_watched(self, lst):
        # FIX: always persist as List[int]
        self.watched_movies = json.dumps([int(x) for x in lst])


# ─────────────────────────── DB INIT ───────────────────────────

def init_db():
    """Create all tables. Safe to call multiple times (no-op if already created)."""
    with app.app_context():
        db.create_all()
        logger.info("Database initialised")


# FIX: create tables at import time so they exist whether the app is run
# directly or served by a WSGI server (gunicorn, uWSGI) that never reaches
# the `if __name__ == '__main__'` block.
init_db()


# ─────────────────────────── HELPERS ───────────────────────────

def current_user():
    uid = session.get('user_id')
    if uid:
        return db.session.get(User, uid)
    return None


def require_login():
    if not session.get('user_id'):
        return jsonify({'error': 'Not authenticated'}), 401
    return None


def ratings_user_id(app_user_id: int) -> int:
    return int(app_user_id) + APP_USER_ID_OFFSET


def get_user_lists_from_ratings(app_user_id: int):
    """
    Derive watched/liked lists from ratings.json.
    Convention:
    - watched: any movie with a rating > 0
    - liked:   rating >= 4.5
    Returns (liked_ids, watched_ids, hist) where IDs are List[int].
    """
    hist = get_user_history(ratings_user_id(app_user_id))
    watched_ids = sorted(hist.keys())
    liked_ids = sorted([mid for mid, r in hist.items() if float(r) >= 4.5])
    return liked_ids, watched_ids, hist


# ─────────────────────────── AUTH ROUTES ───────────────────────────

@app.route('/')
def index():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'All fields required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400

    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password)
    )
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify({'success': True, 'redirect': '/onboarding/movies'})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    identifier = data.get('identifier', '').strip()
    password = data.get('password', '')

    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_id'] = user.id
    if not user.onboarding_done:
        return jsonify({'success': True, 'redirect': '/onboarding/movies'})
    return jsonify({'success': True, 'redirect': '/dashboard'})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


# ─────────────────────────── ONBOARDING ROUTES ───────────────────────────

@app.route('/onboarding/movies')
def onboarding_movies():
    if not session.get('user_id'):
        return redirect('/')
    rec = get_recommender()
    movies = rec.get_popular_movies(n=30)
    return render_template('onboarding_movies.html', movies=movies)


@app.route('/onboarding/preferences')
def onboarding_preferences():
    if not session.get('user_id'):
        return redirect('/')
    rec = get_recommender()
    return render_template('onboarding_preferences.html',
                           genres=rec.get_all_genres(),
                           directors=rec.get_all_directors(),
                           actors=rec.get_all_actors())


@app.route('/api/onboarding/movies', methods=['POST'])
def save_onboarding_movies():
    err = require_login()
    if err:
        return err
    user = current_user()
    data = request.json
    # FIX: cast all incoming IDs to int immediately so set operations work
    # correctly and we never mix int/string types in stored JSON.
    movie_ids = [int(mid) for mid in data.get('movie_ids', [])]
    before, _, _hist = get_user_lists_from_ratings(user.id)
    before_set = set(before)
    liked = list(before_set | set(movie_ids))
    user.set_liked(liked)
    db.session.commit()

    newly_added = set(liked) - before_set
    for mid in newly_added:
        upsert_rating(ratings_user_id(user.id), int(mid), 5.0)
    return jsonify({'success': True})


@app.route('/api/onboarding/preferences', methods=['POST'])
def save_preferences():
    err = require_login()
    if err:
        return err
    user = current_user()
    data = request.json
    user.fav_genres = json.dumps(data.get('genres', []))
    user.fav_directors = json.dumps(data.get('directors', []))
    user.fav_actors = json.dumps(data.get('actors', []))
    user.onboarding_done = True
    db.session.commit()
    return jsonify({'success': True, 'redirect': '/dashboard'})


# ─────────────────────────── MAIN PAGES ───────────────────────────

@app.route('/dashboard')
def dashboard():
    if not session.get('user_id'):
        return redirect('/')
    user = current_user()
    if not user.onboarding_done:
        return redirect('/onboarding/movies')
    return render_template('dashboard.html', username=user.username)


@app.route('/watched')
def watched_page():
    if not session.get('user_id'):
        return redirect('/')
    return render_template('watched.html')


@app.route('/search')
def search_page():
    if not session.get('user_id'):
        return redirect('/')
    return render_template('search.html')


# ─────────────────────────── API ROUTES ───────────────────────────

@app.route('/api/recommendations')
def get_recommendations():
    err = require_login()
    if err:
        return err
    user = current_user()
    rec = get_recommender()

    liked, watched, _hist = get_user_lists_from_ratings(user.id)
    exclude = set(liked + watched)

    mode = (request.args.get('mode') or 'hybrid').strip().lower()
    if mode not in {'content', 'collab', 'hybrid', 'als'}:
        mode = 'hybrid'

    # Use a refresh token to vary results on demand
    t = request.args.get('t')
    try:
        seed = int(t) if t is not None else int(time.time() * 1000)
    except Exception:
        seed = int(time.time() * 1000)
    rng = random.Random(seed)

    candidate_n = 60
    take_n = 12

    recs = []
    if mode in {'als', 'hybrid'}:
        # FIX: use the in-memory cached bundle instead of loading from disk
        bundle = get_als_bundle()
        if bundle is not None:
            ids = als_recommend_for_user(
                bundle, ratings_user_id(user.id),
                n=candidate_n, exclude_item_ids=exclude,
            )
            recs = rec._movies_from_ids(ids)

    if not recs and mode == 'hybrid':
        if len(liked) >= 3:
            recs = rec.recommend(user.id, liked, exclude_ids=exclude, n=candidate_n)
        else:
            recs = rec.recommend_new_user(
                liked, user.get_genres(), user.get_directors(),
                user.get_actors(), exclude_ids=exclude, n=candidate_n,
            )
    elif mode == 'content':
        if liked:
            ids = rec.content_model.recommend(liked, n=candidate_n, exclude_ids=exclude)
            recs = rec._movies_from_ids(ids)
        else:
            recs = rec.recommend_new_user(
                liked, user.get_genres(), user.get_directors(),
                user.get_actors(), exclude_ids=exclude, n=candidate_n,
            )
    elif mode == 'collab':
        ids = rec.collab_model.recommend_for_user(
            user.id, liked_movie_ids=liked, n=candidate_n, exclude_ids=exclude,
        )
        recs = rec._movies_from_ids(ids)

    # Shuffle for refresh variety over the candidate pool
    if recs and len(recs) > take_n:
        top = recs[:candidate_n]
        rng.shuffle(top)
        recs = top[:take_n]
    else:
        recs = recs[:take_n]

    # Fallback to popularity when all recommendation paths return nothing
    if not recs:
        popular_ids = compute_popular_movie_ids(load_ratings(), n=12, exclude_ids=exclude)
        recs = rec._movies_from_ids(popular_ids)

    return jsonify({'recommendations': recs, 'mode': mode})


@app.route('/api/movies/liked')
def get_liked():
    err = require_login()
    if err:
        return err
    user = current_user()
    rec = get_recommender()
    liked_ids, _watched_ids, _hist = get_user_lists_from_ratings(user.id)
    movies = [rec.get_movie(mid) for mid in liked_ids if rec.get_movie(mid)]
    return jsonify({'movies': movies})


@app.route('/api/movies/watched')
def get_watched():
    err = require_login()
    if err:
        return err
    user = current_user()
    rec = get_recommender()
    _liked_ids, watched_ids, _hist = get_user_lists_from_ratings(user.id)
    movies = [rec.get_movie(mid) for mid in watched_ids if rec.get_movie(mid)]
    return jsonify({'movies': movies})


@app.route('/api/movies/like', methods=['POST'])
def like_movie():
    err = require_login()
    if err:
        return err
    user = current_user()
    # FIX: cast to int immediately so all comparisons use a consistent type
    try:
        movie_id = int(request.json.get('movie_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid movie_id'}), 400

    liked, watched, _hist = get_user_lists_from_ratings(user.id)
    if movie_id not in liked:
        liked.append(movie_id)
        user.set_liked(liked)
        db.session.commit()
        upsert_rating(ratings_user_id(user.id), movie_id, 5.0)
    return jsonify({'success': True, 'liked': liked})


@app.route('/api/movies/unlike', methods=['POST'])
def unlike_movie():
    err = require_login()
    if err:
        return err
    user = current_user()
    # FIX: cast to int immediately
    try:
        movie_id = int(request.json.get('movie_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid movie_id'}), 400

    liked, watched, _hist = get_user_lists_from_ratings(user.id)
    if movie_id in liked:
        liked.remove(movie_id)
        user.set_liked(liked)
        db.session.commit()
        delete_rating(ratings_user_id(user.id), movie_id)
    return jsonify({'success': True, 'liked': liked})


@app.route('/api/movies/watch', methods=['POST'])
def mark_watched():
    err = require_login()
    if err:
        return err
    user = current_user()
    # FIX: cast to int immediately
    try:
        movie_id = int(request.json.get('movie_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid movie_id'}), 400

    liked, watched, hist = get_user_lists_from_ratings(user.id)
    if movie_id not in watched:
        watched.append(movie_id)
        user.set_watched(watched)
        db.session.commit()
        # Persist a minimum "watched" rating unless a stronger signal exists
        current = float(hist.get(movie_id, 0.0))
        upsert_rating(ratings_user_id(user.id), movie_id, max(current, 4.0))
    return jsonify({'success': True, 'watched': watched})


@app.route('/api/movies/unwatch', methods=['POST'])
def unmark_watched():
    err = require_login()
    if err:
        return err
    user = current_user()
    # FIX: cast to int immediately
    try:
        movie_id = int(request.json.get('movie_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid movie_id'}), 400

    liked, watched, _hist = get_user_lists_from_ratings(user.id)
    if movie_id in watched:
        watched.remove(movie_id)
        user.set_watched(watched)
        db.session.commit()
        delete_rating(ratings_user_id(user.id), movie_id)
    return jsonify({'success': True, 'watched': watched})


@app.route('/api/movies/rate', methods=['POST'])
def rate_movie():
    err = require_login()
    if err:
        return err
    user = current_user()
    movie_id = request.json.get('movie_id')
    rating = request.json.get('rating')
    try:
        r = float(rating)
        mid = int(movie_id)
    except Exception:
        return jsonify({'error': 'Invalid movie_id or rating'}), 400

    # FIX: threshold must match RATING_MIN from data_loader (0.5) so that a
    # rating submitted below that value triggers deletion rather than being
    # written to disk and then silently ignored on the next load_ratings() call.
    if r < RATING_MIN:
        delete_rating(ratings_user_id(user.id), mid)
    else:
        upsert_rating(ratings_user_id(user.id), mid, r)
    return jsonify({'success': True})


@app.route('/api/movies/search')
def search_movies():
    err = require_login()
    if err:
        return err
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'results': []})
    rec = get_recommender()
    results = rec.search_movies(query)
    user = current_user()
    # FIX: derive liked/watched from ratings.json (the single source of truth)
    # instead of user.get_liked() / user.get_watched() which may lag behind.
    liked_ids, watched_ids, _ = get_user_lists_from_ratings(user.id)
    liked = set(liked_ids)
    watched = set(watched_ids)
    for m in results:
        m['is_liked'] = int(m['id']) in liked
        m['is_watched'] = int(m['id']) in watched
    return jsonify({'results': results})


@app.route('/api/user/status')
def user_status():
    user = current_user()
    if not user:
        return jsonify({'logged_in': False})
    liked, watched, _hist = get_user_lists_from_ratings(user.id)
    return jsonify({
        'logged_in': True,
        'username': user.username,
        'liked_ids': liked,
        'watched_ids': watched,
    })


@app.route('/api/movies/popular')
def popular_movies():
    err = require_login()
    if err:
        return err
    rec = get_recommender()
    user = current_user()
    _liked, watched, _hist = get_user_lists_from_ratings(user.id)
    exclude = set(watched)
    ids = compute_popular_movie_ids(load_ratings(), n=20, exclude_ids=exclude)
    movies = rec._movies_from_ids(ids)
    return jsonify({'movies': movies})


# ─────────────────────────── ENTRY POINT ───────────────────────────

if __name__ == '__main__':
    # Pre-load models so the first request is not slow
    get_recommender()
    if get_als_bundle() is None:
        logger.info("ALS model missing; training from ratings.json...")
        refresh_als_bundle()
    app.run(debug=True, port=5000)
