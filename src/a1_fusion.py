"""
A1 validation-weighted fusion.

The fusion weight is selected exclusively using validation labels.
Test labels must never be passed to choose_validation_weight().
"""

import numpy as np

from sklearn.metrics import f1_score


def fuse_logits(logits_1d, logits_2d, alpha_1d):
    """
    Fuse two models using validation-selected probability weights.

    alpha_1d:
        Weight assigned to the 1-D model.
    1 - alpha_1d:
        Weight assigned to the 2-D model.
    """

    logits_1d = np.asarray(logits_1d, dtype=np.float64)
    logits_2d = np.asarray(logits_2d, dtype=np.float64)

    if logits_1d.shape != logits_2d.shape:
        raise ValueError("Both models must have matching logit shapes.")

    if not 0.0 <= alpha_1d <= 1.0:
        raise ValueError("alpha_1d must be between 0 and 1.")

    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    probabilities_1d = softmax(logits_1d)
    probabilities_2d = softmax(logits_2d)

    return (
        alpha_1d * probabilities_1d
        + (1.0 - alpha_1d) * probabilities_2d
    )


def choose_validation_weight(
    logits_1d,
    logits_2d,
    validation_labels,
):
    """
    Select fusion weight by validation macro-F1.

    Searches alpha from 0 to 1 in increments of 0.05.
    Ties retain the first encountered alpha.
    """

    validation_labels = np.asarray(
        validation_labels,
        dtype=np.int64,
    )

    best_alpha = None
    best_score = -1.0

    for alpha in np.linspace(0.0, 1.0, 21):

        fused = fuse_logits(
            logits_1d,
            logits_2d,
            float(alpha),
        )

        predictions = np.argmax(fused, axis=1)

        score = f1_score(
            validation_labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        if score > best_score:
            best_score = float(score)
            best_alpha = float(alpha)

    return {
        "alpha_1d": best_alpha,
        "alpha_2d": 1.0 - best_alpha,
        "validation_macro_f1": best_score,
    }
