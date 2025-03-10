import spacy

# Load English model or install if not avalible
nlp = spacy.load("en_core_web_sm")


def vector_similarity(ground_truth: str, event_target: str) -> float:
    """
    Computes the vector similarity between two strings using SpaCy.

    Args:
        ground_truth (str): The manually labeled component.
        event_target (str): The inferred counterpart.

    Returns:
        float: Cosine similarity score between the documents.
    """
    doc_gt = nlp(ground_truth)
    doc_et = nlp(event_target)
    return doc_gt.similarity(doc_et)
