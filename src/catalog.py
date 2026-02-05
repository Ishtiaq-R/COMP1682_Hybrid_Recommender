from __future__ import annotations

import pandas as pd

from src.load_movielens import load_movies, load_ratings
from src.load_tmdb_tv import load_tv_shows


_catalog_cache: pd.DataFrame | None = None


def _movie_rating_stats() -> pd.DataFrame:
    ratings = load_ratings()

    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_rating", "count": "rating_count"})
        .reset_index()
    )

    stats["avg_rating"] = pd.to_numeric(stats["avg_rating"], errors="coerce").fillna(0.0)
    stats["rating_count"] = pd.to_numeric(stats["rating_count"], errors="coerce").fillna(0.0)

    return stats


def get_catalog(
    min_movie_ratings: int = 50,
    min_tv_votes: int = 1000,
) -> pd.DataFrame:
    """
    Unified catalog for UI search.

    Returns one DataFrame containing BOTH movies and TV shows
    with a shared schema for the Streamlit UI.
    """
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    # -------------------------
    # MOVIES (MovieLens)
    # -------------------------
    movies = load_movies().copy()
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("Unknown").astype(str)

    stats = _movie_rating_stats()
    movies = movies.merge(stats, on="movieId", how="left")

    movies["avg_rating"] = pd.to_numeric(movies["avg_rating"], errors="coerce").fillna(0.0)
    movies["rating_count"] = pd.to_numeric(movies["rating_count"], errors="coerce").fillna(0.0)

    if min_movie_ratings > 0:
        movies = movies[movies["rating_count"] >= float(min_movie_ratings)]

    movies_out = pd.DataFrame()
    movies_out["content_type"] = "movie"
    movies_out["item_id"] = movies["movieId"].astype(int)
    movies_out["display"] = movies["title"]
    movies_out["title"] = movies["title"]
    movies_out["year"] = ""
    movies_out["genres"] = movies["genres"]
    movies_out["text"] = (
        movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
        + " "
        + movies["genres"].str.replace("|", " ", regex=False)
    )
    movies_out["language"] = ""
    movies_out["popularity"] = 0.0
    movies_out["vote_count"] = movies["rating_count"].astype(float)

    # -------------------------
    # TV SHOWS (TMDB)
    # -------------------------
    tv = load_tv_shows().copy()

    if min_tv_votes > 0 and "vote_count" in tv.columns:
        tv = tv[tv["vote_count"].fillna(0).astype(float) >= float(min_tv_votes)]

    tv_out = pd.DataFrame()
    tv_out["content_type"] = "tv"
    tv_out["item_id"] = tv["tvId"].astype(int)
    tv_out["display"] = tv["display"]
    tv_out["title"] = tv["title"]
    tv_out["year"] = ""
    tv_out["genres"] = tv["genres"]
    tv_out["text"] = tv["text"]
    tv_out["language"] = tv["language"]
    tv_out["popularity"] = tv["popularity"].astype(float)
    tv_out["vote_count"] = tv["vote_count"].astype(float)

    # -------------------------
    # COMBINE
    # -------------------------
    catalog = pd.concat([movies_out, tv_out], ignore_index=True)

    catalog = catalog.dropna(subset=["display"])
    catalog["display"] = catalog["display"].astype(str).str.strip()
    catalog = catalog[catalog["display"].ne("")]

    # Sort so popular titles appear first in dropdown
    catalog["sort_votes"] = pd.to_numeric(catalog["vote_count"], errors="coerce").fillna(0.0)
    catalog = catalog.sort_values(
        ["content_type", "sort_votes", "display"],
        ascending=[True, False, True],
    ).drop(columns=["sort_votes"])

    _catalog_cache = catalog.reset_index(drop=True)
    return _catalog_cache
