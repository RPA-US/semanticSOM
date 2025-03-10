def jaccard_similarity(ground_truth: str, event_target: str) -> float:
    """
    Computes the Jaccard similarity between two strings based on word tokens.

    Args:
        ground_truth (str): The manually labeled component.
        event_target (str): The inferred counterpart.

    Returns:
        float: Jaccard similarity score.
    """
    words_gt = set(ground_truth.lower().split())
    words_et = set(event_target.lower().split())
    intersection = words_gt & words_et
    union = words_gt | words_et
    if not union:
        return 0.0
    return len(intersection) / len(union)
