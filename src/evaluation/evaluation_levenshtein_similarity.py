def levenshtein_distance(s: str, t: str) -> int:
    """
    Computes the Levenshtein distance between two strings.

    Args:
        s (str): First string.
        t (str): Second string.

    Returns:
        int: The Levenshtein distance.
    """
    if len(s) < len(t):
        return levenshtein_distance(t, s)
    if len(t) == 0:
        return len(s)

    previous_row = list(range(len(t) + 1))
    for i, c1 in enumerate(s):
        current_row = [i + 1]
        for j, c2 in enumerate(t):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_similarity(ground_truth: str, event_target: str) -> float:
    """
    Computes a normalized Levenshtein similarity between two strings.

    Args:
        ground_truth (str): The manually labeled component.
        event_target (str): The inferred counterpart.

    Returns:
        float: Normalized similarity score between 0 and 1.
    """
    distance = levenshtein_distance(ground_truth, event_target)
    max_len = max(len(ground_truth), len(event_target))
    if max_len == 0:
        return 1.0
    return 1.0 - distance / max_len
