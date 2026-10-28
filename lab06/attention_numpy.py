"""LAB06: NumPy API implementation; no worked answers to calculation exercises."""
import numpy as np


def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite array')
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """Inputs: (n_query,d_key), (n_key,d_key), (n_key,d_value).

    Returns scaled scores, row-normalized weights, context; no built-in attention.
    """
    Q, K, V = [np.asarray(a, dtype=np.float64) for a in (Q, K, V)]
    if any(a.ndim != 2 or 0 in a.shape for a in (Q, K, V)):
        raise ValueError('Q, K, V must be nonempty matrices')
    if Q.shape[1] != K.shape[1] or K.shape[0] != V.shape[0]:
        raise ValueError('Incompatible query/key dimensions or key/value counts')
    if not all(np.isfinite(a).all() for a in (Q, K, V)):
        raise ValueError('Inputs must be finite')
    scores = Q @ K.T / np.sqrt(K.shape[1])
    attention_weights = stable_softmax(scores, axis=-1)
    context = attention_weights @ V
    return scores, attention_weights, context


def self_attention(X, WQ, WK, WV):
    X, WQ, WK, WV = [np.asarray(a, dtype=np.float64) for a in (X, WQ, WK, WV)]
    Q, K, V = X @ WQ, X @ WK, X @ WV
    scores, weights, context = scaled_dot_product_attention(Q, K, V)
    return Q, K, V, scores, weights, context
