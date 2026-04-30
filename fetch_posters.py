"""
fetch_posters.py
----------------
Updates data/movies.json in-place, replacing placeholder poster URLs
with real ones fetched from the TMDb API.

Usage:
    set TMDB_API_KEY=your_key_here   (Windows)
    export TMDB_API_KEY=your_key     (Mac/Linux)
    python fetch_posters.py

Or pass the key directly:
    python fetch_posters.py --api-key YOUR_KEY
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

MOVIES_PATH = os.path.join("data", "movies.json")
CACHE_PATH  = os.path.join("data", "poster_cache.json")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_PREFIX = "https://picsum.photos"   # detect un-fetched posters


def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def tmdb_search(title: str, year: int, api_key: str) -> str | None:
    """Return the best-match poster URL from TMDb, or None if not found."""
    qs = {
        "api_key": api_key,
        "query": title,
        "include_adult": "false",
    }
    if year:
        qs["year"] = str(year)
    url = "https://api.themoviedb.org/3/search/movie?" + urllib.parse.urlencode(qs)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    [network error] {e}")
        return None

    results = data.get("results") or []
    if not results:
        return None

    # Prefer year-matching results, then sort by popularity
    def _score(r):
        rd = (r.get("release_date") or "")
        ry = int(rd[:4]) if rd[:4].isdigit() else None
        year_match = 1 if (year and ry == year) else 0
        pop = float(r.get("popularity") or 0.0)
        return (year_match, pop)

    results.sort(key=_score, reverse=True)
    poster_path = results[0].get("poster_path")
    if poster_path:
        return TMDB_IMAGE_BASE + poster_path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("TMDB_API_KEY", "").strip())
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even movies that already have a real poster")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No TMDb API key found.")
        print("  Set the TMDB_API_KEY environment variable, or pass --api-key YOUR_KEY")
        return

    with open(MOVIES_PATH, "r", encoding="utf-8") as f:
        movies = json.load(f)

    cache = load_cache()
    updated = 0
    skipped = 0
    failed  = 0

    total = len(movies)
    for i, movie in enumerate(movies, 1):
        title  = movie.get("title", "")
        year   = movie.get("year", 0)
        old_poster = movie.get("poster", "")

        # Skip movies that already have a real poster (unless --force)
        already_real = old_poster and not old_poster.startswith(PLACEHOLDER_PREFIX)
        if already_real and not args.force:
            skipped += 1
            continue

        cache_key = f"{title}::{year}"
        if cache_key in cache and not args.force:
            new_url = cache[cache_key]
        else:
            print(f"[{i}/{total}] Fetching: {title} ({year}) ...", end=" ", flush=True)
            new_url = tmdb_search(title, year, args.api_key)
            time.sleep(0.15)   # stay well within TMDb rate limit (40 req/10s)

            if new_url:
                print(f"OK")
            else:
                print(f"NOT FOUND — keeping placeholder")
                new_url = old_poster   # keep whatever was there

            cache[cache_key] = new_url

        if new_url and new_url != old_poster:
            movie["poster"] = new_url
            updated += 1

    # Save everything
    _atomic_write(MOVIES_PATH, movies)
    _atomic_write(CACHE_PATH, cache)

    print()
    print(f"Done.")
    print(f"  Updated : {updated} posters")
    print(f"  Skipped : {skipped} (already had real poster)")
    print(f"  Failed  : {failed} (kept placeholder)")
    print(f"  Cache   : {CACHE_PATH}")
    print()
    print("Restart your app to see the new posters.")


if __name__ == "__main__":
    main()
