from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_movielens import load_ratings, load_movies

# TMDB enrichment for explanations
try:
    from src.load_tmdb import load_tmdb_metadata
except Exception:  # pragma: no cover
    load_tmdb_metadata = None



_movies_cache: pd.DataFrame | None = None
_ratings_cache: pd.DataFrame | None = None

_vec_cache: TfidfVectorizer | None = None
_X_cache = None
_movieid_to_index_cache: pd.Series | None = None

_rating_stats_cache: pd.DataFrame | None = None

_tmdb_cache: pd.DataFrame | None = None
_tmdb_index_cache: pd.DataFrame | None = None


def _clean_title(s: str) -> str:
    t = str(s)
    t = re.sub(r"\(\d{4}\)", "", t).strip().lower()
    for article in ["the", "a", "an"]:
        suffix = f", {article}"
        if t.endswith(suffix):
            t = f"{article} " + t[: -len(suffix)].strip()
            break
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _ensure_movies() -> pd.DataFrame:
    global _movies_cache
    if _movies_cache is None:
        m = load_movies().copy()
        m["title"] = m["title"].astype(str)
        m["genres"] = m["genres"].fillna("Unknown").astype(str)
        m["clean_title"] = m["title"].apply(_clean_title)
        m["text"] = m["clean_title"] + " " + m["genres"].str.replace("|", " ", regex=False)
        _movies_cache = m
    return _movies_cache


def _ensure_ratings() -> pd.DataFrame:
    global _ratings_cache
    if _ratings_cache is None:
        r = load_ratings().copy()
        _ratings_cache = r
    return _ratings_cache


def _ensure_tfidf():
    global _vec_cache, _X_cache, _movieid_to_index_cache
    movies = _ensure_movies()

    if _vec_cache is None or _X_cache is None or _movieid_to_index_cache is None:
        _vec_cache = TfidfVectorizer(stop_words="english", max_features=50000)
        _X_cache = _vec_cache.fit_transform(movies["text"])

        _movieid_to_index_cache = pd.Series(
            data=np.arange(len(movies), dtype=int),
            index=movies["movieId"].to_numpy(),
        )
    return movies, _X_cache, _movieid_to_index_cache


def _ensure_rating_stats() -> pd.DataFrame:
    global _rating_stats_cache
    if _rating_stats_cache is None:
        r = _ensure_ratings()
        stats = (
            r.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_rating", "count": "rating_count"})
            .reset_index()
        )
        _rating_stats_cache = stats
    return _rating_stats_cache


def _get_tmdb() -> pd.DataFrame | None:
    global _tmdb_cache
    if load_tmdb_metadata is None:
        return None
    if _tmdb_cache is None:
        try:
            _tmdb_cache = load_tmdb_metadata()
        except Exception:
            _tmdb_cache = None
    return _tmdb_cache


def _get_tmdb_index() -> pd.DataFrame | None:
    global _tmdb_index_cache
    tmdb = _get_tmdb()
    if tmdb is None:
        return None

    if _tmdb_index_cache is None:
        tmp = tmdb.copy()
        tmp["norm_title"] = tmp["title"].apply(_clean_title)
        _tmdb_index_cache = tmp.drop_duplicates("norm_title").set_index("norm_title")
    return _tmdb_index_cache


def _user_profile_content_score(
    user_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    X,
    movieid_to_index: pd.Series,
) -> pd.Series:
    ur = user_ratings.copy()
    ur["w"] = ur["rating"] - float(ur["rating"].mean())
    ur = ur[ur["w"] > 0]

    if ur.empty:
        return pd.Series(0.0, index=movies["movieId"].to_numpy())

    mids = ur["movieId"].to_numpy()
    idx = movieid_to_index.reindex(mids).dropna().astype(int).to_numpy()
    if idx.size == 0:
        return pd.Series(0.0, index=movies["movieId"].to_numpy())

    weights_by_movie = ur.set_index("movieId")["w"]
    aligned_mids = movies.iloc[idx]["movieId"].to_numpy()
    weights = weights_by_movie.reindex(aligned_mids).fillna(0.0).to_numpy(dtype=float)

    profile_sparse = X[idx].multiply(weights[:, None]).sum(axis=0)
    profile = np.asarray(profile_sparse)

    sims = cosine_similarity(X, profile).ravel()
    return pd.Series(sims, index=movies["movieId"].to_numpy())


def _cf_style_score(user_ratings: pd.DataFrame, ratings_all: pd.DataFrame):
    liked = user_ratings[user_ratings["rating"] >= 4]["movieId"].unique()
    if liked.size == 0:
        return None, []

    similar_users = ratings_all[ratings_all["movieId"].isin(liked)]["userId"].unique()
    pool = ratings_all[ratings_all["userId"].isin(similar_users)]

    agg = pool.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= 20]

    score = (agg["mean"] - 1.0) / 4.0
    scores = pd.Series(score.to_numpy(dtype=float), index=agg["movieId"].to_numpy())

    anchors = (
        user_ratings[user_ratings["rating"] >= 4]
        .sort_values(["rating", "timestamp"], ascending=[False, False])
        .drop_duplicates(subset=["movieId"])
        .head(5)["movieId"]
        .tolist()
    )
    return scores, anchors


def _build_reason(
    candidate_id: int,
    movies: pd.DataFrame,
    liked_ids: list[int],
    cf_anchor_ids: list[int],
) -> str:
    cand_row = movies.loc[movies["movieId"] == candidate_id]
    if cand_row.empty:
        return "Recommended by your hybrid profile."

    cand = cand_row.iloc[0]
    cand_genres = set(str(cand["genres"]).split("|"))

    parts: list[str] = []

    # TMDB enrichment (optional)
    tmdb_index = _get_tmdb_index()
    if tmdb_index is not None:
        norm = _clean_title(cand["title"])
        if norm in tmdb_index.index:
            row = tmdb_index.loc[norm]
            cast = row["cast"] if isinstance(row.get("cast"), list) else []
            director = row["director"] if isinstance(row.get("director"), list) else []
            keywords = row["keywords"] if isinstance(row.get("keywords"), list) else []

            if cast:
                parts.append("Cast: " + ", ".join([str(x) for x in cast[:2]]))
            if director:
                parts.append("Director: " + ", ".join([str(x) for x in director[:1]]))
            if keywords:
                parts.append("Themes: " + ", ".join([str(x) for x in keywords[:2]]))

    # Shared genres against liked movies
    liked_movies = movies[movies["movieId"].isin(liked_ids)][["movieId", "title", "genres"]].copy()
    if not liked_movies.empty:
        liked_movies["shared"] = liked_movies["genres"].apply(
            lambda g: len(cand_genres.intersection(set(str(g).split("|"))))
        )
        liked_movies = liked_movies.sort_values("shared", ascending=False)
        liked_matches = liked_movies[liked_movies["shared"] > 0].head(2)
    else:
        liked_matches = liked_movies

    shared_genres: set[str] = set()
    liked_titles: list[str] = []
    for _, r in liked_matches.iterrows():
        gset = set(str(r["genres"]).split("|"))
        shared_genres |= cand_genres.intersection(gset)
        liked_titles.append(str(r["title"]))

    shared_genres = {g for g in shared_genres if g and g != "Unknown"}
    if shared_genres:
        parts.append("Shared genres: " + ", ".join(sorted(list(shared_genres))[:2]))
    if liked_titles:
        parts.append("Similar to: " + " and ".join(liked_titles))

    # CF anchors
    if cf_anchor_ids:
        anchor_pool = movies[movies["movieId"].isin(cf_anchor_ids)][["title", "genres"]].copy()
        if not anchor_pool.empty:
            anchor_pool["shared"] = anchor_pool["genres"].apply(
                lambda g: len(cand_genres.intersection(set(str(g).split("|"))))
            )
            anchor_pool = anchor_pool.sort_values("shared", ascending=False)
            anchor_titles = anchor_pool["title"].head(2).astype(str).tolist()
        else:
            anchor_titles = []

        if anchor_titles:
            parts.append("Liked by users who liked: " + " and ".join(anchor_titles))

    if not parts:
        return "Recommended by your hybrid profile."
    return ". ".join(parts)


def _limit_per_genre(df: pd.DataFrame, max_per_genre: int = 3) -> pd.DataFrame:
    counts: dict[str, int] = {}
    keep_rows = []
    for _, row in df.iterrows():
        genres = str(row["genres"]).split("|")
        main = genres[0] if genres else "Unknown"
        counts[main] = counts.get(main, 0) + 1
        if counts[main] <= int(max_per_genre):
            keep_rows.append(row)
    return pd.DataFrame(keep_rows)


def recommend_hybrid(
    user_id: int,
    alpha: float = 0.6,
    top_n: int = 10,
    candidate_pool: int = 300,
    min_ratings: int = 50,
    max_per_genre: int = 3,
) -> pd.DataFrame:
    """
    Hybrid score:
    alpha * CF_style + (1 - alpha) * Content_profile

    Output:
    movieId, title, genres, score, avg_rating, rating_count, reason
    """
    ratings_all = _ensure_ratings()
    movies, X, movieid_to_index = _ensure_tfidf()
    stats = _ensure_rating_stats()

    user_ratings = ratings_all[ratings_all["userId"] == int(user_id)]
    if user_ratings.empty:
        raise ValueError("UserId not found in ratings data.")

    rated_ids = set(user_ratings["movieId"].unique())
    liked_ids = (
        user_ratings[user_ratings["rating"] >= 4]["movieId"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    content_scores = _user_profile_content_score(user_ratings, movies, X, movieid_to_index)

    cf_scores, cf_anchor_ids = _cf_style_score(user_ratings, ratings_all)
    if cf_scores is None:
        cf_scores = pd.Series(0.0, index=movies["movieId"].to_numpy())
        cf_anchor_ids = []
    else:
        cf_scores = cf_scores.reindex(movies["movieId"].to_numpy()).fillna(0.0)

    a = float(alpha)
    a = 0.0 if a < 0 else 1.0 if a > 1 else a

    hybrid = a * cf_scores + (1.0 - a) * content_scores
    hybrid = hybrid[~hybrid.index.isin(rated_ids)]

    # Join rating stats for guardrails + display
    cand_ids = hybrid.sort_values(ascending=False).head(int(max(candidate_pool, top_n))).index.to_list()
    cand = movies[movies["movieId"].isin(cand_ids)][["movieId", "title", "genres"]].copy()
    cand["score"] = cand["movieId"].map(hybrid).astype(float)

    cand = cand.merge(stats, on="movieId", how="left")
    cand["avg_rating"] = pd.to_numeric(cand["avg_rating"], errors="coerce").fillna(0.0)
    cand["rating_count"] = pd.to_numeric(cand["rating_count"], errors="coerce").fillna(0.0)

    if min_ratings and min_ratings > 0:
        cand = cand[cand["rating_count"] >= float(min_ratings)]

    cand = cand.sort_values("score", ascending=False)
    cand = _limit_per_genre(cand, max_per_genre=int(max_per_genre)).head(int(top_n)).copy()

    cand["reason"] = cand["movieId"].apply(
        lambda mid: _build_reason(int(mid), movies, liked_ids, cf_anchor_ids)
    )

    return cand[["movieId", "title", "genres", "score", "avg_rating", "rating_count", "reason"]].reset_index(drop=True)


if __name__ == "__main__":
    df = recommend_hybrid(user_id=1, alpha=0.6, top_n=10)
    print(df.to_string(index=False))

