from pathlib import Path
import ast
import pandas as pd

TMDB_DIR = Path("data/external/tmdb")
TV_FILE = TMDB_DIR / "tmdb_tv_shows.csv"


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # Case-insensitive, whitespace-tolerant column matching
    norm_map = {c.strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in norm_map:
            return norm_map[key]
    return None



def _parse_genres(x) -> list[str]:
    # Handles: list, JSON-like list, comma-separated, pipe-separated, empty
    if isinstance(x, list):
        return [str(g).strip() for g in x if str(g).strip()]

    if not isinstance(x, str):
        return []

    s = x.strip()
    if not s:
        return []

    # JSON-like list string
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, dict) and "name" in item:
                        out.append(str(item["name"]).strip())
                    else:
                        out.append(str(item).strip())
                return [g for g in out if g]
        except Exception:
            pass

    # Fallback split
    if "|" in s:
        parts = s.split("|")
    else:
        parts = s.split(",")

    return [p.strip() for p in parts if p.strip()]


def load_tv_shows() -> pd.DataFrame:
    df = pd.read_csv(TV_FILE)
    lang_col = _first_existing(df, ["original_language", "language", "originalLanguage", "original_lang", "lang"])
    pop_col = _first_existing(df, ["popularity"])
    vc_col = _first_existing(df, ["vote_count", "votes", "num_votes", "voteCount"])
    va_col = _first_existing(df, ["vote_average", "rating", "average_rating", "voteAverage"])
    title_col = _first_existing(df, ["name", "title", "original_name"])
    overview_col = _first_existing(df, ["overview", "description", "plot", "summary"])
    genres_col = _first_existing(df, ["genres", "genre", "genre_names", "genres_name"])
    date_col = _first_existing(df, ["first_air_date", "release_date", "date", "year"])

    if title_col is None:
        raise KeyError(f"No title column found in {TV_FILE.name}. Columns: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["title"] = df[title_col].astype(str).str.strip()

    if overview_col is None:
        out["overview"] = ""
    else:
        out["overview"] = df[overview_col].fillna("").astype(str).str.strip()

    if genres_col is None:
        out["genres_list"] = [[] for _ in range(len(df))]
    else:
        out["genres_list"] = df[genres_col].apply(_parse_genres)

    if date_col is None:
        out["year"] = ""
    else:
        out["year"] = df[date_col].fillna("").astype(str).str.slice(0, 4)

    # Language + popularity columns (optional, if present in the CSV)
    lang_col = _first_existing(df, ["original_language", "language", "originalLanguage", "original_lang", "lang"])
    pop_col = _first_existing(df, ["popularity"])
    vc_col = _first_existing(df, ["vote_count", "votes", "num_votes", "voteCount"])
    va_col = _first_existing(df, ["vote_average", "rating", "average_rating", "voteAverage"])

    # Language
    if lang_col is None:
        out["language"] = ""
    else:
        out["language"] = df[lang_col].fillna("").astype(str).str.lower().str.strip()

    # Popularity signals
    if pop_col is None:
        out["popularity"] = 0.0
    else:
        out["popularity"] = pd.to_numeric(df[pop_col], errors="coerce").fillna(0.0)

    if vc_col is None:
        out["vote_count"] = 0.0
    else:
        out["vote_count"] = pd.to_numeric(df[vc_col], errors="coerce").fillna(0.0)

    if va_col is None:
        out["vote_average"] = 0.0
    else:
        out["vote_average"] = pd.to_numeric(df[va_col], errors="coerce").fillna(0.0)

    out = out[out["title"].ne("")].drop_duplicates(subset=["title"]).reset_index(drop=True)

    out["genres"] = out["genres_list"].apply(lambda xs: "|".join(xs) if xs else "Unknown")
    out["display"] = out["title"] + out["year"].apply(lambda y: f" ({y})" if y and y != "nan" else "")
    out["text"] = (
        out["title"]
        + " "
        + out["genres"].str.replace("|", " ", regex=False)
        + " "
        + out["overview"]
    )

    # Internal ID for TV rows
    out["tvId"] = out.index.astype(int)

    return out[
        ["tvId", "display", "title", "genres", "overview", "text",
         "language", "popularity", "vote_count", "vote_average"]
    ]
