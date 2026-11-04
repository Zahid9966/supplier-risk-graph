"""Model zoo: baselines + gradient boosting + (sim-only) oracle. All take a feature list and expose fit/predict."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F

HGB_GRID = [dict(learning_rate=0.08, max_depth=3, min_samples_leaf=100, max_iter=100, max_bins=64, early_stopping=False),
            dict(learning_rate=0.06, max_depth=4, min_samples_leaf=80, max_iter=150, max_bins=64, early_stopping=False)]


class BaseRate:
    def fit(self, X, y): self.p = float(np.mean(y)); return self
    def predict_proba1(self, X): return np.full(len(X), self.p)


class Logit:
    def __init__(self, C=0.3): self.C = C
    def fit(self, X, y):
        self.m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                               LogisticRegression(C=self.C, max_iter=2000)).fit(X, y); return self
    def predict_proba1(self, X): return self.m.predict_proba(X)[:, 1]


class HGB:
    def __init__(self, seed=0, **kw): self.kw, self.seed = kw, seed
    def fit(self, X, y):
        self.m = HistGradientBoostingClassifier(l2_regularization=1.0, random_state=self.seed, **self.kw).fit(X, y); return self
    def predict_proba1(self, X): return self.m.predict_proba(X)[:, 1]


class Heuristic:
    """No learning: rank by a single column (e.g. propagated risk). Scores are not probabilities."""
    def __init__(self, col): self.col = col
    def fit(self, X, y): return self
    def predict_proba1(self, X): return np.nan_to_num(X[self.col].to_numpy(float))


STATIC, OWN, STRUCT, CONT = F.STATIC_FEATURES, F.OWN_FEATURES, F.GRAPH_STRUCT, F.GRAPH_CONTAGION

# name -> (feature columns, learner kind, description)
SPECS = {
    "base_rate":            ([], "base", "constant prior"),
    "heur_own_composite":   (["own_composite"], "heur:own_composite", "supplier's own distress composite, no learning"),
    "heur_propagated":      (["g_prop_risk"], "heur:g_prop_risk", "noisy-OR propagated composite, no learning"),
    "logit_static":         (STATIC, "logit", "country (World Bank/OFAC) + tier/sector/spend only"),
    "logit_own":            (OWN + STATIC, "logit", "own signals + static, linear"),
    "hgb_own":              (OWN + STATIC, "hgb", "own signals + static, boosted trees (strongest graph-free model)"),
    "hgb_own_struct":       (OWN + STATIC + STRUCT, "hgb", "+ static graph structure (concentration, sole source, criticality)"),
    "hgb_full":             (OWN + STATIC + STRUCT + CONT, "hgb", "+ contagion features (upstream signals propagated through the graph)  <-- MAIN"),
    "logit_full":           (OWN + STATIC + STRUCT + CONT, "logit", "all features, linear"),
}


def make(kind, params=None, seed=0):
    if kind == "base": return BaseRate()
    if kind == "logit": return Logit()
    if kind == "hgb": return HGB(seed=seed, **(params or HGB_GRID[1]))
    if kind.startswith("heur:"): return Heuristic(kind.split(":")[1])
    raise ValueError(kind)
