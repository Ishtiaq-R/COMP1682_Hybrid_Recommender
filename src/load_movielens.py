from pathlib import Path
import pandas as pd


USE_20M = True  

BASE_DIR = Path("data/raw/movielens")

if USE_20M:
    DATA_DIR = BASE_DIR / "ml-20m"
else:
    DATA_DIR = BASE_DIR / "ml-1m"


def load_ratings():
    if USE_20M:
        return pd.read_csv(
            DATA_DIR / "ratings.csv",
            usecols=["userId", "movieId", "rating", "timestamp"]
        )
    else:
        return pd.read_csv(
            DATA_DIR / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"]
        )


def load_movies():
    if USE_20M:
        return pd.read_csv(
            DATA_DIR / "movies.csv",
            usecols=["movieId", "title", "genres"]
        )
    else:
        return pd.read_csv(
            DATA_DIR / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1"
        )


def load_users():
    if USE_20M:
        return None
    else:
        return pd.read_csv(
            DATA_DIR / "users.dat",
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zip"]
        )
