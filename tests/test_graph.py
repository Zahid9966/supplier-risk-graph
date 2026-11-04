import numpy as np
from srg.graph import noisy_or_propagate, pass_weights, descendants_impact


def test_pass_weights_halved_with_alternative():
    assert np.allclose(pass_weights([0.6, 0.6], [False, True]), [0.6, 0.3])


def test_chain_propagation_matches_hand_calc():
    # 2 -> 1 -> 0 (supplier -> customer); node 2 certainly fails, weights 0.5 each
    c, s, w = np.array([1, 0]), np.array([2, 1]), np.array([0.5, 0.5])
    r = noisy_or_propagate(np.array([0.0, 0.0, 1.0]), c, s, w)
    assert np.allclose(r, [0.25, 0.5, 1.0])


def test_noisy_or_two_suppliers():
    c, s, w = np.array([0, 0]), np.array([1, 2]), np.array([0.4, 0.5])
    r = noisy_or_propagate(np.array([0.1, 0.5, 0.2]), c, s, w)
    assert np.isclose(r[0], 1 - 0.9 * (1 - 0.4 * 0.5) * (1 - 0.5 * 0.2))


def test_properties_bounds_and_monotonicity():
    rng = np.random.default_rng(0); n = 40
    c = rng.integers(0, n, 120); s = rng.integers(0, n, 120); w = rng.uniform(0, 1, 120)
    s0 = rng.uniform(0, 0.5, n); r = noisy_or_propagate(s0, c, s, w)
    assert ((r >= s0 - 1e-12) & (r <= 1)).all()                      # propagation never lowers risk, stays a probability
    r2 = noisy_or_propagate(np.minimum(s0 + 0.2, 1), c, s, w)
    assert (r2 >= r - 1e-12).all()                                    # monotone in inputs (also on cyclic graphs)


def test_zero_weights_and_shock():
    c, s = np.array([0, 1]), np.array([1, 2])
    assert np.allclose(noisy_or_propagate(np.array([0.1, 0.2, 0.3]), c, s, np.zeros(2)), [0.1, 0.2, 0.3])
    r = descendants_impact(3, c, s, np.array([1.0, 0.5]), [2])
    assert np.allclose(r, [0.5, 0.5, 1.0])
