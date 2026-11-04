"""Train the production model, score the latest month, and write serving artifacts (gold/)."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd

from . import config, evaluate as E, models as M, features as F
from .graph import build_w, descendants_impact, noisy_or_propagate


def write_serving_artifacts(silver, g, ev=None):
    out = config.layer("gold")
    lab = g[g.y.notna()]
    cols = M.SPECS["hgb_full"][0]
    gi = (ev or {}).get("chosen_grid_index", {}).get("hgb_full")
    gi = 1 if gi is None else gi
    model = M.make("hgb", M.HGB_GRID[gi]).fit(lab[cols], lab.y)
    # calibrator: fitted on validation predictions of the train-only model (same recipe as the evaluation)
    sp = E.split(g); tr, va = sp["train"], sp["val"]
    m0 = M.make("hgb", M.HGB_GRID[gi]).fit(tr[cols], tr.y)
    cal = E.platt_fit(m0.predict_proba1(va[cols]), va.y.to_numpy())
    joblib.dump(dict(model=model, calibrator=cal, features=cols, grid_index=gi), out / "model.joblib")

    t_last = int(g.t.max()); cur = g[g.t == t_last].reset_index(drop=True)
    sup = silver["suppliers"].set_index("supplier_id")
    raw = model.predict_proba1(cur[cols]); p = E.platt_apply(cal, raw)
    neutral = cur.copy(); med = lab[M.CONT].median()
    for c in M.CONT: neutral[c] = med[c]
    p_nc = E.platt_apply(cal, model.predict_proba1(neutral[cols]))

    ids = cur.supplier_id.tolist(); idx = {s: i for i, s in enumerate(ids)}
    p0 = silver["signals"].period.min(); last_period = p0 + t_last
    ed = silver["edges"]; ed = ed[ed.valid_from_month <= last_period].copy()
    ed["pass_weight"] = ed.share * np.where(ed.alt_source, 0.5, 1.0)
    s0 = np.nan_to_num(cur.own_composite.to_numpy())
    ed["driver"] = ed.pass_weight * ed.from_supplier.map(lambda s: s0[idx[s]])
    drivers = (ed.sort_values("driver", ascending=False).groupby("to_supplier").head(3)
                 .groupby("to_supplier").apply(lambda d: json.dumps([dict(supplier_id=a, contribution=round(float(b), 4)) for a, b in zip(d.from_supplier, d.driver)]), include_groups=False))
    scores = pd.DataFrame(dict(supplier_id=ids, month=cur.month, p_disruption_3m=p, p_raw=raw, p_without_graph_contagion=p_nc,
                               graph_uplift=p - p_nc, own_composite=cur.own_composite, propagated_risk=cur.g_prop_risk,
                               criticality=cur.g_criticality, n_suppliers=cur.g_n_suppliers.astype(int), sole_source=cur.g_sole_source.astype(bool),
                               months_since_incident=cur.months_since_inc))
    scores = scores.join(sup[["name", "country_iso3", "sector", "tier"]], on="supplier_id")
    scores["top_upstream_drivers"] = scores.supplier_id.map(drivers).fillna("[]")
    scores["rank"] = scores.p_disruption_3m.rank(ascending=False, method="first").astype(int)
    scores.sort_values("rank").to_parquet(out / "scores.parquet", index=False)
    ed[["from_supplier", "to_supplier", "share", "alt_source", "pass_weight"]].to_parquet(out / "graph_edges.parquet", index=False)
    sup.reset_index().to_parquet(out / "graph_nodes.parquet", index=False)

    bom = silver["bom"].merge(scores[["supplier_id", "p_disruption_3m"]], on="supplier_id")
    bom["exposed"] = bom.bom_share * bom.p_disruption_3m
    prod = bom.groupby("product").agg(annual_revenue_usd=("annual_revenue_usd", "first"), n_suppliers=("supplier_id", "size"),
                                      prob_any_supplier_disrupted=("p_disruption_3m", lambda x: float(1 - np.prod(1 - x))),
                                      expected_share_exposed=("exposed", "sum")).reset_index()
    prod["expected_revenue_exposed_usd"] = prod.annual_revenue_usd * prod.expected_share_exposed
    prod.to_parquet(out / "products.parquet", index=False)
    return dict(as_of=str(last_period), suppliers=len(scores), mean_p=float(p.mean()), top=scores.nsmallest(5, "rank")[["supplier_id", "p_disruption_3m"]].to_dict("records"))
