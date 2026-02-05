from __future__ import annotations

import pandas as pd

from src.load_movielens import load_ratings, load_movies


def popular_movies(
    n: int = 10,
    min_ratings: int = 200,
) -> pd.DataFrame:
    """
    Popular movie baseline using MovieLens.
    Works for ml-1m or ml-20m via load_movielens.USE_20M.

    Returns:
    title, genres, mean, count
    """
    ratings = load_ratings()
    movies = load_movies()

    agg = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    agg["mean"] = pd.to_numeric(agg["mean"], errors="coerce").fillna(0.0)
    agg["count"] = pd.to_numeric(agg["count"], errors="coerce").fillna(0.0)

    agg = agg[agg["count"] >= float(min_ratings)]
    agg = agg.sort_values(["mean", "count"], ascending=[False, False])

    out = agg.merge(movies, on="movieId", how="left").head(int(n))

    return out[["title", "genres", "mean", "count"]].reset_index(drop=True)


if __name__ == "__main__":
    df = popular_movies(n=10, min_ratings=200)
    print(df.to_string(index=False))
