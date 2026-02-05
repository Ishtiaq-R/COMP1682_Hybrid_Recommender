from __future__ import annotations

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_tmdb_tv import load_tv_shows


_tv_cache: pd.DataFrame | None = None
_vec_cache: TfidfVectorizer | None = None
_X_cache = None


def _norm_lang(x: str) -> str:
    s = str(x).strip().lower()
    if s == "english":
        return "en"
    return s


def _strip_year(s: str) -> str:
    # "Breaking Bad (2008)" -> "Breaking Bad"
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(s)).strip()


def _ensure_tv_and_tfidf():
    global _tv_cache, _vec_cache, _X_cache

    if _tv_cache is None:
        _tv_cache = load_tv_shows()

    if _vec_cache is None or _X_cache is None:
        _vec_cache = TfidfVectorizer(stop_words="english", max_features=50000)
        _X_cache = _vec_cache.fit_transform(_tv_cache["text"])

    return _tv_cache, _X_cache


def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn = float(s.min())
    mx = float(s.max())
    if mx - mn < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def _make_reason(src: pd.Series, rec: pd.Series) -> str:
    src_genres = set(str(src.get("genres", "")).split("|"))
    rec_genres = set(str(rec.get("genres", "")).split("|"))
    shared = sorted([g for g in (src_genres & rec_genres) if g and g != "Unknown"])[:2]

    parts: list[str] = []
    if shared:
        parts.append("Shared genres: " + ", ".join(shared))

    if str(rec.get("language", "")).strip():
        parts.append("Same language")

    vc = float(rec.get("vote_count", 0) or 0)
    if vc >= 5000:
        parts.append("Widely rated")
    elif vc >= 1000:
        parts.append("Good rating volume")

    if not parts:
        return "Similar description and genre profile"
    return ". ".join(parts)


def recommend_tv_similar(
    display_query: str,
    top_n: int = 10,
    only_same_language: bool = True,
    min_vote_count: int = 1000,
    w_sim: float = 0.80,
    w_pop: float = 0.15,
    w_votes: float = 0.05,
) -> pd.DataFrame:
    tv, X = _ensure_tv_and_tfidf()

    dq = str(display_query).strip()

    # 1) Exact display match
    exact = tv[tv["display"] == dq]
    if not exact.empty:
        idx = int(exact.index[0])
    else:
        # 2) Display contains match
        mask = tv["display"].str.contains(dq, case=False, na=False, regex=False)
        if mask.any():
            idx = int(tv[mask].index[0])
        else:
            # 3) Title fallback (removes year)
            dq_title = _strip_year(dq)
            mask_t = tv["title"].str.contains(dq_title, case=False, na=False, regex=False)
            if not mask_t.any():
                raise ValueError("No matching TV show found. Try a different search.")
            idx = int(tv[mask_t].index[0])

    sims = cosine_similarity(X[idx], X).ravel()

    cand = tv.copy()
    cand["sim"] = sims
    cand = cand[cand["display"] != tv.loc[idx, "display"]]

    if only_same_language and "language" in cand.columns and "language" in tv.columns:
        src_lang = _norm_lang(tv.loc[idx, "language"])
        cand = cand[cand["language"].apply(_norm_lang) == src_lang]

    if "vote_count" in cand.columns and min_vote_count and min_vote_count > 0:
        cand = cand[cand["vote_count"].fillna(0).astype(float) >= float(min_vote_count)]

    pop = cand["popularity"].fillna(0.0).astype(float) if "popularity" in cand.columns else pd.Series(0.0, index=cand.index)
    votes = cand["vote_count"].fillna(0.0).astype(float) if "vote_count" in cand.columns else pd.Series(0.0, index=cand.index)

    pop_n = _norm01(pop)
    votes_n = _norm01(votes)

    w_sum = float(w_sim + w_pop + w_votes)
    if w_sum <= 0:
        w_sim, w_pop, w_votes = 1.0, 0.0, 0.0
        w_sum = 1.0

    w_sim /= w_sum
    w_pop /= w_sum
    w_votes /= w_sum

    cand["score"] = float(w_sim) * cand["sim"] + float(w_pop) * pop_n + float(w_votes) * votes_n

    out = cand.sort_values("score", ascending=False).head(int(top_n)).copy()
    src_row = tv.loc[idx]
    out["reason"] = out.apply(lambda r: _make_reason(src_row, r), axis=1)

    return out[["tvId", "display", "genres", "overview", "score", "reason"]].reset_index(drop=True)
