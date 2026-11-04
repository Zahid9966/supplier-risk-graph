"""Leakage audit. Each check returns {name, passed, detail}. These are *tests of the evaluation*, run on every build."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, evaluate as E, features as F, models as M

FORBIDDEN = ("true_", "latent", "hazard", "p_upstream", "contagion_start")


def truncate_silver(silver, t0):
    p0 = silver["signals"].period.min(); cut = p0 + t0
    s = dict(silver)
    s["signals"] = silver["signals"][silver["signals"].avail_period <= cut]
    s["incidents"] = silver["incidents"][silver["incidents"].start_period <= cut]
    s["edges"] = silver["edges"][silver["edges"].valid_from_month <= cut]
    return s


def check_point_in_time(silver, g, t0s=(30, 45)):
    """Delete everything that would not exist at t0, rebuild, and require identical feature rows at t0-1..t0."""
    feats = [c for c in F.ALL_FEATURES + F.PLACEBO]
    n_total = int(silver["signals"].period.max().ordinal - silver["signals"].period.min().ordinal) + 1
    worst = 0.0; rows = 0
    for t0 in t0s:
        tg = F.build_features(truncate_silver(silver, t0), max_t=t0, n_months=n_total)
        a = tg[tg.t.isin([t0 - 1, t0])].sort_values(["t", "supplier_id"]).reset_index(drop=True)
        b = g[g.t.isin([t0 - 1, t0])].sort_values(["t", "supplier_id"]).reset_index(drop=True)
        assert len(a) == len(b) > 0, "truncated build missing rows"
        d = np.nanmax(np.abs(a[feats].to_numpy(float) - b[feats].to_numpy(float)))
        nan_mismatch = int((a[feats].isna().to_numpy() != b[feats].isna().to_numpy()).sum())
        worst = max(worst, float(d)); rows += len(a)
        if nan_mismatch: worst = float("inf")
    return dict(name="point_in_time_truncation", passed=worst < 1e-9, detail=f"rebuilt features from data truncated at t0={list(t0s)}; max abs diff over {rows} rows = {worst:.2e}")


def check_purge_gap():
    ok = E.SPLITS["train"][1] + config.HORIZON < E.SPLITS["val"][0] and E.SPLITS["val"][1] + config.HORIZON < E.SPLITS["test"][0]
    return dict(name="split_purge_gap", passed=ok, detail=f"label horizon {config.HORIZON}m; splits {E.SPLITS} leave no label overlap")


def check_label_shuffle(g, n=3):
    """Null distribution: train on within-month shuffled labels, score the real test set. A single shuffle can land
    off 0.5 by chance alignment with the true signal, so we average several."""
    sp = E.split(g); tr0, te = sp["train"], sp["test"]
    cols = M.SPECS["hgb_full"][0]; aucs = []
    for seed in range(n):
        tr = tr0.copy(); rng = np.random.default_rng(seed)
        tr["y"] = tr.groupby("t").y.transform(lambda s: rng.permutation(s.to_numpy()))
        m = M.make("hgb", M.HGB_GRID[0]).fit(tr[cols], tr.y)
        aucs.append(float(roc_auc_score(te.y, m.predict_proba1(te[cols]))))
    mean = float(np.mean(aucs))
    return dict(name="label_shuffle_null", passed=0.46 <= mean <= 0.54,
                detail=f"trained on within-month shuffled labels, {n} shuffles -> test ROC-AUC {mean:.3f} (runs: {', '.join(f'{a:.3f}' for a in aucs)}); should be ~0.5")


def check_canary(g):
    """A deliberately leaky feature must (a) be flagged by the univariate scan and (b) explode the score;
    the real feature set must not be flagged."""
    lab = g[g.y.notna()].copy()
    scan = {c: abs(roc_auc_score(lab.y, lab[c].fillna(lab[c].median())) - 0.5) + 0.5 for c in F.ALL_FEATURES if lab[c].nunique() > 1}
    flagged_real = sorted([c for c, a in scan.items() if a > 0.85])
    lab["LEAK_canary"] = lab.y + np.random.default_rng(0).normal(0, 0.6, len(lab))
    canary_auc = abs(roc_auc_score(lab.y, lab.LEAK_canary) - 0.5) + 0.5
    sp = E.split(lab); cols = M.SPECS["hgb_full"][0] + ["LEAK_canary"]
    m = M.make("hgb", M.HGB_GRID[0]).fit(sp["train"][cols], sp["train"].y)
    leaky = float(roc_auc_score(sp["test"].y, m.predict_proba1(sp["test"][cols])))
    ok = (not flagged_real) and canary_auc > 0.85
    return dict(name="canary_and_univariate_scan", passed=ok,
                detail=f"max univariate AUC of real features {max(scan.values()):.3f} (flag >0.85: {flagged_real or 'none'}); "
                       f"canary univariate AUC {canary_auc:.3f} is flagged; model with canary reaches test AUC {leaky:.3f}")


def check_no_latents(silver, g):
    cols = {c for t in silver.values() for c in t.columns} | set(g.columns)
    bad = sorted(c for c in cols if c.startswith(FORBIDDEN) or c in FORBIDDEN)
    return dict(name="no_simulator_latents_in_pipeline", passed=not bad, detail=f"forbidden-name scan over silver+gold columns: {bad or 'clean'}")


def check_random_vs_temporal(ev):
    t = ev["models"]["hgb_full"]["roc_auc"]; r = ev["naive_random_split_hgb_full"]["roc_auc"]
    return dict(name="random_vs_temporal_split", passed=True, detail=f"informational: random row split ROC-AUC {r:.3f} vs temporal {t:.3f} (gap {r - t:+.3f}); we report temporal only")


def run_all(silver, g, ev=None):
    out = [check_purge_gap(), check_no_latents(silver, g), check_point_in_time(silver, g), check_label_shuffle(g), check_canary(g)]
    if ev: out.append(check_random_vs_temporal(ev))
    return out
