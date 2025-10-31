import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def predict_topics(
    content_embedding: np.ndarray,
    topic_embeddings: np.ndarray,
    topic_ids: List[str],
    min_score: float = 0.3,
    top_k: int = 3
) -> List[str]:
    """
    Predict topics using embedding similarity only.

    Returns up to `top_k` topic ids whose cosine similarity with the content
    embedding is >= min_score.
    """
    # Calculate embedding similarities
    embedding_similarities = cosine_similarity(
        content_embedding.reshape(1, -1),
        topic_embeddings
    )[0]

    # Use embeddings-only scores
    hybrid_scores = embedding_similarities

    # Get top-k topics above threshold
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    predicted_topics = [
        topic_ids[idx] for idx in top_indices
        if hybrid_scores[idx] >= min_score
    ]

    return predicted_topics
