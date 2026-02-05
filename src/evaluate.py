import pandas as pd

from src.load_movielens import load_ratings, load_movies
from src.split import train_test_split_per_user
from src.eval_metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.recommend_hybrid import recommend_hybrid

def relevant_items(test_df: pd.DataFrame, user_id: int, threshold: float = 4.0) -> set[int]:
    u = test_df[(test_df["userId"] == user_id) & (test_df["rating"] >= threshold)]
    return set(u["movieId"].to_list())

def evaluate_hybrid(test_df: pd.DataFrame, k: int = 10, alpha: float = 0.6) -> dict:
    users = sorted(test_df["userId"].unique().tolist())
    p_list, r_list, n_list = [], [], []

    for uid in users:
        rel = relevant_items(test_df, uid)
        if len(rel) == 0:
            continue

        rec_df = recommend_hybrid(user_id=int(uid), alpha=float(alpha), top_n=max(k, 20))
        recs = rec_df["movieId"].to_list()

        p_list.append(precision_at_k(recs, rel, k))
        r_list.append(recall_at_k(recs, rel, k))
        n_list.append(ndcg_at_k(recs, rel, k))

    if len(p_list) == 0:
        return {"precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0, "users_scored": 0}

    return {
        "precision@k": round(sum(p_list) / len(p_list), 4),
        "recall@k": round(sum(r_list) / len(r_list), 4),
        "ndcg@k": round(sum(n_list) / len(n_list), 4),
        "users_scored": len(p_list),
    }

def main():
    ratings = load_ratings()
    _ = load_movies()

    train_df, test_df = train_test_split_per_user(ratings, test_size=5, seed=42)



    for alpha in [0.2, 0.4, 0.6, 0.8]:
        result = evaluate_hybrid(test_df, k=10, alpha=alpha)
        print("alpha =", alpha, result)

if __name__ == "__main__":
    main()
