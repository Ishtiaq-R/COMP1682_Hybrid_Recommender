import numpy as np

def precision_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = recs[:k]
    if len(topk) == 0:
        return 0.0
    hits = sum(1 for x in topk if x in relevant)
    return hits / float(k)

def recall_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    topk = recs[:k]
    hits = sum(1 for x in topk if x in relevant)
    return hits / float(len(relevant))

def ndcg_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    topk = recs[:k]
    if len(topk) == 0:
        return 0.0

    dcg = 0.0
    for i, mid in enumerate(topk, start=1):
        rel = 1.0 if mid in relevant else 0.0
        dcg += rel / np.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg
