from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_movielens import load_movies, load_ratings



_movies_cache: pd.DataFrame | None = None
_vec_cache: TfidfVectorizer | None = None
_X_cache = None

_rating_stats_cache: pd.DataFrame | None = None


def _clean_title(s: str) -> str:
    t = str(s)
    t = re.sub(r"\(\d{4}\)", "", t).strip().lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _ensure_movies() -> pd.DataFrame:
    global _movies_cache
    if _movies_cache is None:
        m = load_movies().copy()
        m["title"] = m["title"].astype(str)
        m["clean_title"] = m["title"].apply(_clean_title)
        m["genres"] = m["genres"].fillna("Unknown").astype(str)

        m["text"] = (
            m["clean_title"]
            + " "
            + m["genres"].str.replace("|", " ", regex=False)
        )

        _movies_cache = m
    return _movies_cache


def _ensure_tfidf():
    global _vec_cache, _X_cache
    movies = _ensure_movies()

    if _vec_cache is None or _X_cache is None:
        _vec_cache = TfidfVectorizer(stop_words="english", max_features=50000)
        _X_cache = _vec_cache.fit_transform(movies["text"])
    return movies, _X_cache


def _ensure_rating_stats() -> pd.DataFrame:
    global _rating_stats_cache
    if _rating_stats_cache is None:
        r = load_ratings()
        agg = (
            r.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_rating", "count": "rating_count"})
            .reset_index()
        )
        _rating_stats_cache = agg
    return _rating_stats_cache


def _norm01(x: pd.Series) -> pd.Series:
    a = pd.to_numeric(x, errors="coerce").fillna(0.0)
    mn = float(a.min())
    mx = float(a.max())
    if mx - mn < 1e-9:
        return pd.Series(0.0, index=a.index)
    return (a - mn) / (mx - mn)


def _build_reason(
    seed_genres: set[str],
    rec_title: str,
    rec_genres: str,
    avg_rating: float | None,
    rating_count: float | None,
    med_avg: float,
    med_cnt: float,
) -> str:
    gset = set(str(rec_genres).split("|")) if rec_genres else set()
    shared = sorted([g for g in (seed_genres & gset) if g and g != "Unknown"])[:2]

    parts: list[str] = []
    if shared:
        parts.append("Shared genres: " + ", ".join(shared))

    if avg_rating is not None and float(avg_rating) >= med_avg:
        parts.append("High average rating")

    if rating_count is not None and float(rating_count) >= med_cnt:
        parts.append("Popular with many ratings")

    if not parts:
        return "Similar title and genre profile"
    return ". ".join(parts)


def recommend_similar_title(
    title_query: str,
    top_n: int = 10,
    min_ratings: int = 50,
    w_sim: float = 0.70,
    w_pop: float = 0.20,
    w_votes: float = 0.10,
) -> pd.DataFrame:
    """
    Content similarity (TF-IDF) plus popularity guardrails.

    Returns columns:
    movieId, title, genres, score, avg_rating, rating_count, reason
    """
    movies, X = _ensure_tfidf()
    stats = _ensure_rating_stats()

    q = _clean_title(title_query)

    # Find seed movie
    mask = movies["clean_title"].str.contains(q, case=False, na=False, regex=False)
    if not mask.any():
        raise ValueError("No matching title found. Try a different search.")

    seed_idx = int(movies[mask].index[0])
    seed_row = movies.loc[seed_idx]
    seed_movie_id = int(seed_row["movieId"])
    seed_genres = set(str(seed_row["genres"]).split("|"))

    sims = cosine_similarity(X[seed_idx], X).ravel()

    cand = movies[["movieId", "title", "genres"]].copy()
    cand["sim"] = sims
    cand = cand[cand["movieId"] != seed_movie_id]
    # Family friendly auto-filter
    seed_g = set(str(seed_row["genres"]).split("|"))
    family_seed = bool(seed_g.intersection({"Animation", "Children", "Family"}))

    if family_seed:
        allow = {"Animation", "Children", "Family", "Adventure", "Comedy", "Fantasy"}
        block = {"Horror", "Crime", "Thriller", "War", "Mystery"}

        def _ok(genres: str) -> bool:
            g = set(str(genres).split("|"))
            if g.intersection(block):
                return False
            return bool(g.intersection(allow))

        cand = cand[cand["genres"].apply(_ok)]


    cand = cand.merge(stats, on="movieId", how="left")
    cand["avg_rating"] = pd.to_numeric(cand["avg_rating"], errors="coerce").fillna(0.0)
    cand["rating_count"] = pd.to_numeric(cand["rating_count"], errors="coerce").fillna(0.0)

    # Guardrail: remove low-signal items
    if min_ratings and min_ratings > 0:
        cand = cand[cand["rating_count"] >= float(min_ratings)]

    # Popularity components
    avg_n = _norm01(cand["avg_rating"])
    cnt_n = _norm01(np.log1p(cand["rating_count"]))

    w_sum = float(w_sim + w_pop + w_votes)
    if w_sum <= 0:
        w_sim, w_pop, w_votes = 1.0, 0.0, 0.0
        w_sum = 1.0

    w_sim /= w_sum
    w_pop /= w_sum
    w_votes /= w_sum

    cand["score"] = (
        float(w_sim) * cand["sim"]
        + float(w_pop) * avg_n
        + float(w_votes) * cnt_n
    )

    med_avg = float(cand["avg_rating"].median()) if len(cand) else 0.0
    med_cnt = float(cand["rating_count"].median()) if len(cand) else 0.0

    out = cand.sort_values("score", ascending=False).head(int(top_n)).copy()
    out["reason"] = out.apply(
        lambda r: _build_reason(
            seed_genres=seed_genres,
            rec_title=str(r["title"]),
            rec_genres=str(r["genres"]),
            avg_rating=float(r["avg_rating"]),
            rating_count=float(r["rating_count"]),
            med_avg=med_avg,
            med_cnt=med_cnt,
        ),
        axis=1,
    )

    cols = ["movieId", "title", "genres", "score", "avg_rating", "rating_count", "reason"]
    return out[cols].reset_index(drop=True)


if __name__ == "__main__":
    df = recommend_similar_title("Toy Story", top_n=10)
    print(df.to_string(index=False))
