from sentence_transformers import SentenceTransformer, util

# Load SBERT model globally
model = SentenceTransformer("all-mpnet-base-v2")


def sbert_similarity(ground_truth: str, event_target: str) -> float:
    """
    Computes the cosine similarity between embeddings of two strings using SBERT.

    Args:
        ground_truth (str): The manually labeled component.
        event_target (str): The inferred counterpart.

    Returns:
        float: Cosine similarity score between the embeddings.
    """
    if "<error>" in event_target:
        return 0.0
    embedding_gt = model.encode(ground_truth)
    embedding_et = model.encode(event_target)
    similarity_score = util.pytorch_cos_sim(embedding_gt, embedding_et).item()
    return similarity_score


if __name__ == "__main__":
    print(
        sbert_similarity(
            "go to testing api documentation button", "testing api documentation button"
        )
    )
    print(sbert_similarity("toggle night mode button", "call initiation button"))
    print(
        sbert_similarity(
            "Convert file to Microsoft powerpoint",
            "powerpoint file conversion initiation",
        )
    )
    print(sbert_similarity("Change theme primary color", "primary color configuration"))
    print(
        sbert_similarity(
            "Change question status to resolved", "question status update to resolved"
        )
    )
    print(
        sbert_similarity(
            "change border radius", "ui customization - border radius adjustment"
        )
    )
    print(
        sbert_similarity(
            "search for openssh software", "software search (e.g., 'openssh' query)"
        )
    )
    print(
        sbert_similarity(
            "create exam calendar event", "schedule calendar event with notification"
        )
    )
