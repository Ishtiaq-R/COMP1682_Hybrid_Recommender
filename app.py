import streamlit as st

from src.catalog import get_catalog
from src.recommend_content import recommend_similar_title
from src.recommend_tv import recommend_tv_similar
from src.recommend_hybrid import recommend_hybrid

def render_recommendations(df, header: str):
    st.subheader(header)

    if df is None or df.empty:
        st.info("No recommendations found.")
        return

    # pick a display column
    title_col = "display" if "display" in df.columns else ("title" if "title" in df.columns else df.columns[0])

    for i, row in df.reset_index(drop=True).iterrows():
        title = str(row.get(title_col, "Unknown title"))
        score = row.get("score", None)

        label = f"{i+1}. {title}"
        if score is not None:
            try:
                label += f"  |  score {float(score):.3f}"
            except Exception:
                pass

        with st.expander(label, expanded=False):
            reason = str(row.get("reason", "")).strip()
            if reason:
                st.write("Reason")
                st.write(reason)

            genres = str(row.get("genres", "")).strip()
            if genres:
                st.write("Genres")
                st.write(genres.replace("|", ", "))

            overview = str(row.get("overview", "")).strip()
            if overview:
                st.write("Overview")
                st.write(overview)

            # optional ids if you want them visible
            if "tvId" in df.columns:
                st.write(f"tvId: {row.get('tvId')}")
            if "movieId" in df.columns:
                st.write(f"movieId: {row.get('movieId')}")

@st.cache_data
def cached_catalog():
    return get_catalog()


st.set_page_config(page_title="Hybrid Movie & TV Recommender", layout="centered")

st.title("Hybrid Movie & TV Recommender")
st.write("Search a title, pick it, then get recommendations with reasons.")

# Cache reset (helps after catalog.py changes)
if st.button("Reset search cache"):
    st.cache_data.clear()
    st.rerun()

catalog = cached_catalog()

top_n = st.slider("Number of recommendations", 5, 20, 10)
show_scores = st.checkbox("Show technical scores", value=False)

query = st.text_input("Search for a movie or TV show")

options = []
option_map = {}

if query:
    q = str(query).strip()
    mask = catalog["display"].str.contains(q, case=False, na=False, regex=False)
    subset = catalog.loc[mask].head(60)

    for _, row in subset.iterrows():
        ctype = str(row["content_type"])
        item_id = int(row["item_id"])
        display = str(row["display"])

        label = display
        options.append(label)
        option_map[label] = (ctype, item_id, display)

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



if st.button("Show recommendations") and selected:
    ctype, _, display = option_map[selected]

    if ctype == "movie":
        w_sim = 0.90 - 0.30 * prefer_popular
        w_pop = 0.05 + 0.20 * prefer_popular
        w_votes = 0.05 + 0.10 * prefer_popular

        with st.spinner("Generating recommendations..."):
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

        render_recommendations(df, "Similar movies")

    else:
        with st.spinner("Generating recommendations..."):
            df = recommend_tv_similar(
                display_query=str(display),
                top_n=int(top_n),
            )

        if not show_scores and "score" in df.columns:
            df = df.drop(columns=["score"])

        render_recommendations(df, "Similar TV shows")

with st.expander("Advanced: MovieLens evaluation mode"):
    st.write("Uses MovieLens userId. Use this for evaluation screenshots.")

    user_id = st.number_input("MovieLens userId", min_value=1, value=1, step=1)
    alpha = st.slider("Alpha", 0.0, 1.0, 0.6)

    if st.button("Run MovieLens hybrid"):
        with st.spinner("Generating recommendations..."):
            df = recommend_hybrid(
                user_id=int(user_id),
                alpha=float(alpha),
                top_n=int(top_n),
            )

        if not show_scores and "score" in df.columns:
            df = df.drop(columns=["score"])

        render_recommendations(df, "Hybrid recommendations")
