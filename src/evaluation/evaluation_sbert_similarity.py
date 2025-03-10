from sentence_transformers import SentenceTransformer, util

# Load SBERT model globally
model = SentenceTransformer("all-MiniLM-L6-v2")


def sbert_similarity(ground_truth: str, event_target: str) -> float:
    """
    Computes the cosine similarity between embeddings of two strings using SBERT.

    Args:
        ground_truth (str): The manually labeled component.
        event_target (str): The inferred counterpart.

    Returns:
        float: Cosine similarity score between the embeddings.
    """
    embedding_gt = model.encode(ground_truth)
    embedding_et = model.encode(event_target)
    similarity_score = util.pytorch_cos_sim(embedding_gt, embedding_et).item()
    return similarity_score
