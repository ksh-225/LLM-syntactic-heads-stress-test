"""
A.6 — Fix 2: Syntactic Head Isolation.

For each (layer, head) pair, compute the fraction of tokens where the argmax
attention points to the correct UD head. Retain only (layer, head) pairs
exceeding random_baseline + sigma * std as genuine "syntactic heads".
"""
import numpy as np


def identify_syntactic_heads(all_conf_matrices, n_layers, n_heads, sigma=2.0):
    """
    all_conf_matrices: list of [n_layers, n_heads] argmax-correctness matrices,
                        one per evaluated token.

    Returns:
        heads: list of (layer, head) tuples above threshold
        acc: [n_layers, n_heads] accuracy matrix
        threshold: scalar cutoff used
    """
    acc = np.zeros((n_layers, n_heads))
    for mat in all_conf_matrices:
        best = np.unravel_index(np.argmax(mat), mat.shape)
        acc[best] += 1
    acc /= len(all_conf_matrices)

    baseline = 1.0 / (n_layers * n_heads)
    threshold = baseline + sigma * np.std(acc)

    heads = [
        (l, h)
        for l in range(n_layers)
        for h in range(n_heads)
        if acc[l, h] > threshold
    ]
    return heads, acc, threshold
