import streamlit as st

from src.catalog import get_catalog
from src.recommend_content import recommend_similar_title
from src.recommend_hybrid import recommend_hybrid
from src.recommend_tv import recommend_tv_similar


@st.cache_data
def cached_catalog():
    return get_catalog()


st.set_page_config(page_title="Hybrid Movie & TV Recommender", layout="centered")

st.title("Hybrid Movie & TV Recommender (With enhanced reasoning)")
st.write("Search once, pick titles you like, then get recommendations with reasons.")

catalog = cached_catalog()

mode = st.radio(
    "Recommendation mode",
    [
        "Similar titles",
        "Personalised by liked movies",
        "Personalised by MovieLens userId",
    ],
    index=0,
)

top_n = st.slider("Number of recommendations", 5, 20, 10)
show_scores = st.checkbox("Show technical scores", value=False)


def build_options(query: str, limit: int = 40):
    options = []
    option_map = {}

    if not query:
        return options, option_map

    q = str(query).strip()
    mask = catalog["display"].str.contains(q, case=False, na=False, regex=False)
    subset = catalog.loc[mask].head(limit)

    for _, row in subset.iterrows():
        ctype = str(row["content_type"])
        item_id = int(row["item_id"])
        display = str(row["display"])

        label = ("Movie | " if ctype == "movie" else "TV | ") + display
        options.append(label)
        option_map[label] = (ctype, item_id, display)

    return options, option_map


# -------------------------
# MODE 1: SIMILAR TITLES
# -------------------------
if mode == "Similar titles":
    query = st.text_input("Search for a movie or TV show")
    options, option_map = build_options(query)

    selected = st.selectbox("Select a title", options) if options else None

    pop_choice = st.select_slider(
        "Popularity level",
        options=["Niche", "Popular", "Blockbusters"],
        value="Popular",
    )

    prefer_popular_map = {
        "Niche": 0.3,
        "Popular": 0.6,
        "Blockbusters": 0.9,
    }
    prefer_popular = float(prefer_popular_map[pop_choice])
    only_english_tv = st.checkbox("Only English", value=True)

    if st.button("Show recommendations") and selected:
        ctype, _, display = option_map[selected]

        if ctype == "movie":
            w_sim = 0.90 - 0.30 * float(prefer_popular)
            w_pop = 0.05 + 0.20 * float(prefer_popular)
            w_votes = 0.05 + 0.10 * float(prefer_popular)

            df = recommend_similar_title(
                title_query=str(display),
                top_n=int(top_n),
                min_ratings=50,
                w_sim=float(w_sim),
                w_pop=float(w_pop),
                w_votes=float(w_votes),
            )

            if not show_scores:
                drop_cols = [c for c in ["score", "avg_rating", "rating_count"] if c in df.columns]
                if drop_cols:
                    df = df.drop(columns=drop_cols)

            st.subheader("Similar movies")
            st.dataframe(df, use_container_width=True)

        else:
            df = recommend_tv_similar(
                display_query=str(display),
                top_n=int(top_n),
                only_same_language=bool(only_english_tv),
            )

            if not show_scores and "score" in df.columns:
                df = df.drop(columns=["score"])

            st.subheader("Similar TV shows")
            st.dataframe(df, use_container_width=True)


# -------------------------
# MODE 2: PERSONALISED BY LIKED MOVIES
# -------------------------
elif mode == "Personalised by liked movies":
    st.write("Pick a few movies you like. The system will combine them into one profile.")

    query = st.text_input("Search movies to add")
    options, option_map = build_options(query)

    # Only allow movie picks for this mode
    movie_options = [o for o in options if o.startswith("Movie | ")]

    picked = st.multiselect("Liked movies", movie_options)

    prefer_popular = st.slider("Prefer popular results", 0.0, 1.0, 0.6)

    if st.button("Show recommendations"):
        if not picked:
            st.error("Add at least one movie first.")
        else:
            # Simple and effective: merge multiple rec lists and re-rank
            w_sim = 0.90 - 0.30 * float(prefer_popular)
            w_pop = 0.05 + 0.20 * float(prefer_popular)
            w_votes = 0.05 + 0.10 * float(prefer_popular)

            frames = []
            for label in picked:
                _, _, display = option_map[label]
                df = recommend_similar_title(
                    title_query=str(display),
                    top_n=50,
                    min_ratings=50,
                    w_sim=float(w_sim),
                    w_pop=float(w_pop),
                    w_votes=float(w_votes),
                )
                frames.append(df)

            all_recs = (
                st.session_state.get("tmp_df")
            )

            merged = None
            if frames:
                merged = frames[0]
                for f in frames[1:]:
                    merged = merged.merge(
                        f[["movieId", "score"]],
                        on="movieId",
                        how="outer",
                        suffixes=("", "_r"),
                    )

                score_cols = [c for c in merged.columns if c == "score" or c.endswith("_r")]
                merged["score_merged"] = merged[score_cols].fillna(0.0).mean(axis=1)

                merged = merged.sort_values("score_merged", ascending=False)

                # remove the picked items themselves
                picked_titles = {option_map[x][2] for x in picked}
                merged = merged[~merged["title"].isin(picked_titles)]

                merged = merged.head(int(top_n)).copy()
                merged["reason"] = "Matches your liked movies profile"

                out = merged[["movieId", "title", "genres", "score_merged", "reason"]].rename(
                    columns={"score_merged": "score"}
                )

                if not show_scores and "score" in out.columns:
                    out = out.drop(columns=["score"])

                st.subheader("Personalised movie recommendations")
                st.dataframe(out, use_container_width=True)


# -------------------------
# MODE 3: PERSONALISED BY MOVIELENS USERID
# -------------------------
else:
    st.write("This mode is for evaluation and testing with MovieLens users.")

    user_id = st.number_input("MovieLens userId", min_value=1, value=1, step=1)

    alpha = st.slider(
        "Balance between user behaviour and content similarity",
        0.0,
        1.0,
        0.6,
    )

    if st.button("Show recommendations"):
        df = recommend_hybrid(
            user_id=int(user_id),
            alpha=float(alpha),
            top_n=int(top_n),
        )

        if not show_scores and "score" in df.columns:
            df = df.drop(columns=["score"])

        st.subheader("Personalised movie recommendations")
        st.dataframe(df, use_container_width=True)
