"""Risk propagation on the dependency graph.

pass-through weight  w_ij = share_ij * (0.5 if an alternative source exists else 1)   (supplier j -> customer i)
noisy-OR propagation r_i = 1 - (1 - s_i) * prod_j (1 - w_ij * r_j), iterated to a fixed point (graph may be cyclic).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def pass_weights(share, alt):
    return np.asarray(share, float) * np.where(np.asarray(alt, bool), 0.5, 1.0)


def build_w(n, cust_idx, sup_idx, w):
    """Sparse (customer x supplier) matrix."""
    return sp.csr_matrix((w, (cust_idx, sup_idx)), shape=(n, n))


def noisy_or_propagate(s, cust_idx, sup_idx, w, iters=8, tol=1e-9):
    """Effective risk after propagation. s: own risk in [0,1]. Returns r (same shape)."""
    s = np.clip(np.nan_to_num(np.asarray(s, float)), 0, 1)
    r = s.copy()
    for _ in range(iters):
        logq = np.zeros_like(s)
        np.add.at(logq, cust_idx, np.log1p(-np.minimum(w * r[sup_idx], 0.999999)))
        new = 1 - (1 - s) * np.exp(logq)
        if np.max(np.abs(new - r)) < tol:
            r = new; break
        r = new
    return r


def descendants_impact(n, cust_idx, sup_idx, w, shocked, iters=8):
    """What-if: force `shocked` nodes to risk 1 and propagate; returns effective risk of every node."""
    s = np.zeros(n); s[list(shocked)] = 1.0
    r = noisy_or_propagate(s, cust_idx, sup_idx, w, iters=iters)
    return r
