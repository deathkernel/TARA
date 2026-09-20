"""Training metrics for TARA."""


def accuracy(predictions, targets, threshold=0.5):
    """Binary accuracy using a configurable threshold."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("accuracy() requires at least one sample")
    correct = 0
    for prediction, target in zip(predictions, targets):
        predicted_label = 1 if prediction >= threshold else 0
        target_label = 1 if target >= threshold else 0
        correct += predicted_label == target_label
    return correct / len(predictions)
