def recall_at_k(recommended_ids, ground_truth_ids, k):
    recommended_top_k = recommended_ids[:k]
    hits = set(recommended_top_k) & set(ground_truth_ids)
    return 1.0 if len(hits) > 0 else 0.0


def mrr_at_k(recommended_ids, ground_truth_ids, k):
    for rank, job_id in enumerate(recommended_ids[:k], start=1):
        if job_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0