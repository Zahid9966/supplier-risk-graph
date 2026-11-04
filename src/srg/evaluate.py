"""Honest evaluation: temporal purged splits, baselines, ablations, clustered bootstrap CIs, calibration, cold-start."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config, features as F, models as M

H = config.HORIZON
# months 0..59; labels observable for t <= 56.  Train labels reach t+H=38 < val start; val labels reach 49 < test start.
SPLITS = dict(train=(config.WARMUP, 35), val=(39, 46), test=(50, 56))


def split(g, T=None):
    lab = g[g.y.notna()]
    out = {k: lab[(lab.t >= a) & (lab.t <= b)] for k, (a, b) in SPLITS.items()}
    assert SPLITS["train"][1] + H < SPLITS["val"][0] and SPLITS["val"][1] + H < SPLITS["test"][0], "label horizon overlaps next split"
    return out


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop", labels=False) if len(np.unique(p)) > bins else pd.Series(np.zeros(len(p), int))
    d = pd.DataFrame(dict(y=y, p=p, b=np.asarray(q)))
    a = d.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float((a.n * (a.y - a.p).abs()).sum() / a.n.sum())


def topk_metrics(df, score, frac=0.05):
    """Pooled over test months: precision and recall of the top `frac` per month (what an analyst queue would see)."""
    d = df[["t", "y"]].assign(s=score)
    hits = pos = sel = 0
    for _, m in d.groupby("t"):
        k = max(1, int(round(frac * len(m)))); top = m.nlargest(k, "s")
        hits += top.y.sum(); sel += k; pos += m.y.sum()
    return dict(precision_at_5pct=float(hits / sel), recall_at_5pct=float(hits / max(pos, 1)), lift_at_5pct=float((hits / sel) / (pos / len(d))))


def metrics(df, score, calibrated=True):
    y = df.y.to_numpy()
    out = dict(roc_auc=float(roc_auc_score(y, score)), pr_auc=float(average_precision_score(y, score)), **topk_metrics(df, score))
    if calibrated:
        out.update(brier=float(brier_score_loss(y, score)), ece=ece(y, score))
    return out


def cluster_bootstrap(df, scores: dict, n=200, seed=0, ref=None):
    """Resample SUPPLIERS (clusters) with replacement; CIs for PR-AUC and paired PR-AUC differences vs `ref`."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("supplier_id").indices; keys = np.array(list(groups)); y = df.y.to_numpy()
    res = {k: [] for k in scores}; diff = {k: [] for k in scores}
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        if y[idx].sum() == 0: continue
        pr = {k: average_precision_score(y[idx], s[idx]) for k, s in scores.items()}
        for k in scores:
            res[k].append(pr[k])
            if ref: diff[k].append(pr[k] - pr[ref])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: dict(pr_auc_ci=ci(res[k]), **({"pr_auc_diff_vs_" + ref: dict(mean=float(np.mean(diff[k])), ci=ci(diff[k]),
            frac_positive=float(np.mean(np.array(diff[k]) > 0)))} if ref and k != ref else {})) for k in scores}


def fit_select(spec, tr, va, seed=0):
    """Pick HGB hyper-params on validation PR-AUC (same budget for every HGB variant)."""
    cols, kind, _ = spec
    if kind != "hgb":
        return M.make(kind, seed=seed).fit(tr[cols] if cols else tr, tr.y), None
    best = None
    for i, params in enumerate(M.HGB_GRID):
        m = M.make(kind, params, seed).fit(tr[cols], tr.y)
        sc = average_precision_score(va.y, m.predict_proba1(va[cols]))
        if best is None or sc > best[0]: best = (sc, i, m)
    return best[2], best[1]


def platt_fit(p_val, y_val):
    z = np.log(np.clip(p_val, 1e-6, 1 - 1e-6) / (1 - np.clip(p_val, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_val); return lr


def platt_apply(lr, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def run_evaluation(g: pd.DataFrame, oracle: pd.DataFrame | None = None, n_boot=200, quick=False):
    sp = split(g); tr, va, te = sp["train"], sp["val"], sp["test"]
    specs = dict(M.SPECS)
    results, scores, chosen, fitted = {}, {}, {}, {}
    if oracle is not None:      # SIMULATOR-ONLY upper bound: uses latent truth that no real system could observe
        g = g.merge(oracle, on=["t", "supplier_id"], how="left")
        sp = split(g); tr, va, te = sp["train"], sp["val"], sp["test"]
        specs["ORACLE_sim_only"] = (M.OWN + M.STATIC + M.STRUCT + M.CONT + ["true_p_upstream", "true_active", "true_fin_stress", "true_u"], "hgb",
                                    "uses simulator latents - an unattainable ceiling, NOT a model")
    for name, spec in specs.items():
        if quick and name in ("logit_full", "hgb_own_struct"): continue
        model, gi = fit_select(spec, tr, va)
        cols = spec[0]
        raw = model.predict_proba1(te[cols] if cols else te)
        scores[name] = raw; chosen[name] = gi; fitted[name] = model
        results[name] = dict(desc=spec[2], features=len(cols), **metrics(te, raw, calibrated=spec[1] != "heur" and not spec[1].startswith("heur")))
    ci = cluster_bootstrap(te.reset_index(drop=True), scores, n=n_boot, ref="hgb_own")
    for k in results: results[k].update(ci[k])
    # head-to-head vs the strongest graph-free model on the non-graph metrics, paired
    ci2 = cluster_bootstrap(te.reset_index(drop=True), {"hgb_own_struct": scores["hgb_own_struct"], "hgb_full": scores["hgb_full"]} if not quick else {"hgb_full": scores["hgb_full"], "hgb_own": scores["hgb_own"]}, n=n_boot, ref="hgb_own_struct" if not quick else "hgb_own")

    # calibration of the main model: Platt scaling fitted on validation only
    main_cols = M.SPECS["hgb_full"][0]
    mm, gi_main = fitted["hgb_full"], chosen["hgb_full"]
    cal = platt_fit(mm.predict_proba1(va[main_cols]), va.y.to_numpy())
    p_raw = mm.predict_proba1(te[main_cols]); p_cal = platt_apply(cal, p_raw)
    calibration = dict(raw=dict(brier=float(brier_score_loss(te.y, p_raw)), ece=ece(te.y.to_numpy(), p_raw)),
                       platt_on_val=dict(brier=float(brier_score_loss(te.y, p_cal)), ece=ece(te.y.to_numpy(), p_cal)),
                       base_rate_brier=float(brier_score_loss(te.y, np.full(len(te), tr.y.mean()))))

    # ablations: drop one feature group at a time from the main model, add placebo
    groups = {"own_signals": M.OWN, "static_country_sector": M.STATIC, "graph_structure": M.STRUCT, "graph_contagion": M.CONT}
    abl = {}
    full_cols = M.SPECS["hgb_full"][0]
    for gname, gcols in groups.items():
        cols = [c for c in full_cols if c not in gcols]
        m = M.make("hgb", M.HGB_GRID[gi_main]).fit(tr[cols], tr.y); p = m.predict_proba1(te[cols])
        abl["minus_" + gname] = metrics(te, p, calibrated=False)
    cols = full_cols + F.PLACEBO
    m = M.make("hgb", M.HGB_GRID[gi_main]).fit(tr[cols], tr.y)
    abl["plus_placebo_downstream"] = metrics(te, m.predict_proba1(te[cols]), calibrated=False)
    abl["full"] = results["hgb_full"] | {}
    # feature-group permutation importance on test for the main model (drop in PR-AUC when shuffled within month)
    rng = np.random.default_rng(0); imp = {}
    base_pr = average_precision_score(te.y, p_raw)
    for gname, gcols in groups.items():
        drops = []
        for _ in range(5):
            Xp = te[main_cols].copy()
            perm = rng.permutation(len(Xp))
            for c in gcols:
                if c in Xp: Xp[c] = Xp[c].to_numpy()[perm]
            drops.append(base_pr - average_precision_score(te.y, mm.predict_proba1(Xp)))
        imp[gname] = float(np.mean(drops))

    # cold-start: hold out 20% of suppliers entirely from training
    rng = np.random.default_rng(1); sids = g.supplier_id.unique(); hold = set(rng.choice(sids, int(0.2 * len(sids)), replace=False))
    trc, vac, tec = (d[~d.supplier_id.isin(hold)] for d in (tr, va, te)); tec_h = te[te.supplier_id.isin(hold)]
    cold = {}
    for name in ("hgb_own", "hgb_full"):
        spec = M.SPECS[name]; m = M.make("hgb", M.HGB_GRID[chosen[name]]).fit(trc[spec[0]], trc.y)
        cold[name] = metrics(tec_h, m.predict_proba1(tec_h[spec[0]]), calibrated=False)

    # naive random split (for the leakage section: how much does ignoring time flatter the model?)
    lab = g[g.y.notna() & (g.t >= config.WARMUP)]
    rs = np.random.default_rng(2).random(len(lab)) < 0.7
    spec = M.SPECS["hgb_full"]; m = M.make("hgb", M.HGB_GRID[gi_main]).fit(lab[rs][spec[0]], lab[rs].y)
    random_split = metrics(lab[~rs], m.predict_proba1(lab[~rs][spec[0]]), calibrated=False)

    return dict(
        splits={k: dict(t_range=list(v), rows=int(len(sp[k])), positives=int(sp[k].y.sum()), base_rate=float(sp[k].y.mean())) for k, v in SPLITS.items()},
        horizon_months=H, models=results, head_to_head_full_vs_struct=ci2, chosen_grid_index=chosen,
        calibration_main_model=calibration, ablations=abl, group_permutation_importance_pr_auc_drop=imp,
        cold_start_20pct_suppliers_held_out=dict(rows=int(len(tec_h)), positives=int(tec_h.y.sum()), **cold),
        naive_random_split_hgb_full=random_split)
