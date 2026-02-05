import pandas as pd
import ast
from pathlib import Path

TMDB_DIR = Path("data/external/tmdb")

def _pick_title_column(df: pd.DataFrame) -> str:
    cols = [c.strip() for c in df.columns]
    df.columns = cols  # strip spaces

    if "title" in df.columns:
        return "title"
    if "original_title" in df.columns:
        return "original_title"
    if "movie_title" in df.columns:
        return "movie_title"

    raise KeyError(f"No title column found. Columns are: {df.columns.tolist()}")

def load_tmdb_metadata():
    movies = pd.read_csv(TMDB_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(TMDB_DIR / "tmdb_5000_credits.csv")

    # strip column whitespace
    movies.columns = [c.strip() for c in movies.columns]
    credits.columns = [c.strip() for c in credits.columns]

    title_col = _pick_title_column(movies)

    # parse JSON-like columns
    if "genres" in movies.columns:
        movies["genres"] = movies["genres"].apply(
            lambda x: [g.get("name") for g in ast.literal_eval(x)] if isinstance(x, str) else []
        )
    else:
        movies["genres"] = [[] for _ in range(len(movies))]

    if "keywords" in movies.columns:
        movies["keywords"] = movies["keywords"].apply(
            lambda x: [k.get("name") for k in ast.literal_eval(x)] if isinstance(x, str) else []
        )
    else:
        movies["keywords"] = [[] for _ in range(len(movies))]

    # credits format
    if "cast" in credits.columns:
        credits["cast"] = credits["cast"].apply(
            lambda x: [c.get("name") for c in ast.literal_eval(x)[:5]] if isinstance(x, str) else []
        )
    else:
        credits["cast"] = [[] for _ in range(len(credits))]

    if "crew" in credits.columns:
        credits["director"] = credits["crew"].apply(
            lambda x: [c.get("name") for c in ast.literal_eval(x) if c.get("job") == "Director"] if isinstance(x, str) else []
        )
    else:
        credits["director"] = [[] for _ in range(len(credits))]

    # merge on tmdb id
    # movies uses "id", credits uses "movie_id" (usually)
    if "id" not in movies.columns:
        raise KeyError(f"'id' missing in tmdb_5000_movies.csv. Columns: {movies.columns.tolist()}")
    if "movie_id" not in credits.columns:
        raise KeyError(f"'movie_id' missing in tmdb_5000_credits.csv. Columns: {credits.columns.tolist()}")

    meta = movies.merge(credits[["movie_id", "cast", "director"]], left_on="id", right_on="movie_id", how="left")

    meta["title"] = meta[title_col].astype(str).str.strip()

    return meta[["title", "genres", "keywords", "cast", "director"]]
