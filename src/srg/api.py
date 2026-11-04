"""FastAPI service over the gold layer. Read-only; scores are produced offline by `srg score`."""
from __future__ import annotations
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, __version__
from .graph import descendants_impact

app = FastAPI(title="supplier-risk-graph", version=__version__,
              description="Multi-tier supplier disruption risk. All supplier data is SIMULATED; see /about.")
DASH = config.ROOT / "dashboard"


@lru_cache(maxsize=1)
def store():
    g = config.data_root() / "gold"
    if not (g / "scores.parquet").exists():
        raise HTTPException(503, "gold artifacts missing - run `python -m srg.cli run` first")
    scores = pd.read_parquet(g / "scores.parquet").set_index("supplier_id", drop=False)
    edges = pd.read_parquet(g / "graph_edges.parquet")
    idx = {s: i for i, s in enumerate(scores.index)}
    ci = edges.to_supplier.map(idx).to_numpy(); si = edges.from_supplier.map(idx).to_numpy()
    return dict(scores=scores, edges=edges, idx=idx, ci=ci, si=si, w=edges.pass_weight.to_numpy(),
                products=pd.read_parquet(g / "products.parquet"))


def reset_cache():
    store.cache_clear()


@app.get("/health")
def health():
    s = store()
    return {"status": "ok", "suppliers": len(s["scores"]), "edges": len(s["edges"]), "as_of": str(s["scores"].month.iloc[0])}


@app.get("/about")
def about():
    return {"data": "simulated supply network + real World Bank/OFAC country covariates",
            "horizon": f"{config.HORIZON} months", "caveat": "scores demonstrate a method on synthetic data, not real-world accuracy"}


@app.get("/suppliers")
def suppliers(limit: int = Query(50, ge=1, le=500), tier: int | None = None, country: str | None = None, min_p: float = 0.0):
    d = store()["scores"]
    if tier: d = d[d.tier == tier]
    if country: d = d[d.country_iso3 == country.upper()]
    d = d[d.p_disruption_3m >= min_p].sort_values("rank").head(limit)
    return json.loads(d.drop(columns=["p_raw"]).to_json(orient="records"))


@app.get("/suppliers/{sid}")
def supplier(sid: str):
    s = store(); sid = sid.upper()
    if sid not in s["idx"]: raise HTTPException(404, "unknown supplier")
    row = json.loads(s["scores"].loc[[sid]].drop(columns=["p_raw"]).to_json(orient="records"))[0]
    row["top_upstream_drivers"] = json.loads(row["top_upstream_drivers"])
    e = s["edges"]
    row["suppliers"] = json.loads(e[e.to_supplier == sid].to_json(orient="records"))
    row["customers"] = json.loads(e[e.from_supplier == sid].to_json(orient="records"))
    return row


@app.get("/graph/{sid}")
def subgraph(sid: str, depth: int = Query(2, ge=1, le=4)):
    """Upstream neighbourhood (suppliers of suppliers ...) up to `depth` hops plus direct customers."""
    s = store(); sid = sid.upper()
    if sid not in s["idx"]: raise HTTPException(404, "unknown supplier")
    e = s["edges"]; keep, frontier = {sid}, {sid}
    for _ in range(depth):
        nxt = set(e[e.to_supplier.isin(frontier)].from_supplier) - keep
        keep |= nxt; frontier = nxt
    cust = set(e[e.from_supplier == sid].to_supplier); keep |= cust
    sub = e[e.from_supplier.isin(keep) & e.to_supplier.isin(keep)]
    nodes = s["scores"].loc[sorted(keep), ["supplier_id", "name", "tier", "country_iso3", "p_disruption_3m", "propagated_risk"]]
    return {"center": sid, "nodes": json.loads(nodes.to_json(orient="records")), "edges": json.loads(sub[["from_supplier", "to_supplier", "share", "pass_weight"]].to_json(orient="records"))}


class WhatIf(BaseModel):
    shocked: list[str] = Field(min_length=1, max_length=50, description="suppliers assumed fully disrupted")


@app.post("/simulate")
def what_if(req: WhatIf):
    """Deterministic what-if: force `shocked` suppliers to risk 1 and propagate (noisy-OR) down the dependency graph."""
    s = store(); ids = [x.upper() for x in req.shocked]
    bad = [x for x in ids if x not in s["idx"]]
    if bad: raise HTTPException(404, f"unknown supplier(s): {bad}")
    r = descendants_impact(len(s["idx"]), s["ci"], s["si"], s["w"], [s["idx"][x] for x in ids])
    d = s["scores"][["supplier_id", "name", "tier"]].copy(); d["impact"] = r
    d = d[~d.supplier_id.isin(ids) & (d.impact > 0.01)].sort_values("impact", ascending=False)
    prod = s["products"].copy()
    bom = pd.read_parquet(config.data_root() / "silver" / "bom.parquet")
    bom["impact"] = bom.supplier_id.map(dict(zip(s["scores"].supplier_id, r)))
    pi = bom.groupby("product").apply(lambda x: float((x.bom_share * x.impact).sum()), include_groups=False)
    prod["what_if_share_impacted"] = prod["product"].map(pi).fillna(0); prod["what_if_revenue_at_risk_usd"] = prod.annual_revenue_usd * prod.what_if_share_impacted
    return {"shocked": ids, "n_impacted": int(len(d)), "impacted": json.loads(d.head(50).to_json(orient="records")),
            "products": json.loads(prod.sort_values("what_if_revenue_at_risk_usd", ascending=False).to_json(orient="records")),
            "note": "impact = propagated disruption probability under the pass-through model; not a calibrated forecast"}


@app.get("/products")
def products():
    return json.loads(store()["products"].sort_values("expected_revenue_exposed_usd", ascending=False).to_json(orient="records"))


@app.get("/metrics")
def metrics():
    p = config.reports_root() / "metrics.json"
    if not p.exists(): raise HTTPException(404, "no evaluation report; run `python -m srg.cli evaluate`")
    return json.loads(p.read_text())


@app.get("/leakage")
def leakage_report():
    p = config.reports_root() / "leakage.json"
    if not p.exists(): raise HTTPException(404, "no leakage report")
    return json.loads(p.read_text())


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(DASH / "index.html")


app.mount("/static", StaticFiles(directory=DASH), name="static")
