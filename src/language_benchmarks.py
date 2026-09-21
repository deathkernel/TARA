"""Small, dependency-free language-model evaluation metrics for TARA.

Research basis:
- Autoregressive language models are trained to predict the next token.
- Perplexity is the exponentiated mean negative log-likelihood.
- Next-token top-1 accuracy is a complementary direct prediction metric.

These metrics are intentionally simple. They are useful for regression
tracking on TARA's tiny experiments, not as a substitute for broad benchmark
suites used to compare large language models.
"""

import math


def evaluate(model, dataset):
    """Evaluate mean loss, perplexity, top-1 accuracy, and token count."""
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    for inputs, targets in dataset.iter_windows():
        logits = model.forward(inputs)
        if len(logits) != len(targets):
            raise ValueError("model output length does not match targets")
        for values, target in zip(logits, targets):
            scores = [value.data for value in values]
            maximum = max(scores)
            total_loss += maximum + math.log(
                sum(math.exp(score - maximum) for score in scores)
            ) - scores[target]
            if max(range(len(scores)), key=scores.__getitem__) == target:
                correct += 1
            total_tokens += 1

    if total_tokens == 0:
        raise ValueError("dataset must contain at least one target token")

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "accuracy": correct / total_tokens,
        "tokens": total_tokens,
    }
