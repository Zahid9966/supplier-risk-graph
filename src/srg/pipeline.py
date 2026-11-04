"""Bronze (raw, immutable, lineage-tagged) -> Silver (typed, cleaned, deduplicated, quarantined) layers."""
from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd

from . import config
from .dq import DQ
from .simulate import COUNTRY_VARIANTS

RAW_FILES = ["suppliers_erp.csv", "supply_edges.csv", "signals_monthly.jsonl", "incidents.csv", "products_bom.csv"]

# country name/ISO2 variants -> ISO3 (superset of the variants the simulator injects, plus common real-world ones)
_ISO2 = {"CN": "CHN", "VN": "VNM", "US": "USA", "DE": "DEU", "IN": "IND", "MX": "MEX", "TR": "TUR", "KR": "KOR",
         "JP": "JPN", "GB": "GBR", "UK": "GBR", "PL": "POL", "TH": "THA", "MY": "MYS", "ID": "IDN", "BR": "BRA"}
_NAMES = {"china": "CHN", "prc": "CHN", "viet nam": "VNM", "vietnam": "VNM", "united states": "USA", "u.s.a.": "USA",
          "deutschland": "DEU", "germany": "DEU", "india": "IND", "mexico": "MEX", "méxico": "MEX", "turkey": "TUR",
          "türkiye": "TUR", "south korea": "KOR", "korea, rep.": "KOR", "japan": "JPN"}


def to_iso3(x, known):
    if not isinstance(x, str):
        return None
    s = x.strip()
    if s.upper() in known:
        return s.upper()
    return _ISO2.get(s.upper()) or _NAMES.get(s.lower())


# ------------------------------------------------------------------ bronze
def run_bronze(raw_dir=None, run_ts=None):
    raw_dir = raw_dir or config.layer("raw"); out = config.layer("bronze")
    run_ts = run_ts or pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {}
    for fn in RAW_FILES:
        p = raw_dir / fn
        df = pd.read_json(p, lines=True, dtype=False) if fn.endswith(".jsonl") else pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
        if fn.endswith(".jsonl"):
            df = df.astype({c: "object" for c in df.columns})
        df["_source_file"] = fn
        df["_ingest_run"] = run_ts
        df["_row_hash"] = pd.util.hash_pandas_object(df.drop(columns=["_ingest_run"]).astype(str), index=False).astype("uint64")
        df.to_parquet(out / (fn.rsplit(".", 1)[0] + ".parquet"), index=False)
        manifest[fn] = dict(rows=len(df), sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    (out / "_manifest.json").write_text(json.dumps(dict(run=run_ts, files=manifest), indent=2))
    return manifest


# ------------------------------------------------------------------ silver
def _derive_tier(ids, edges):
    """Tier = 1 + shortest hops to a node with no outgoing edge (those only supply the buyer)."""
    out = edges.groupby("from_supplier").to_supplier.apply(list).to_dict()
    tier = {}
    order = list(ids)
    for _ in range(8):
        for s in order:
            outs = out.get(s, [])
            tier[s] = 1 if not outs else 1 + min(tier.get(c, 99) for c in outs)
    return pd.Series(tier)


def run_silver():
    b, s = config.layer("bronze"), config.layer("silver")
    ind = pd.read_csv(config.EXTERNAL / "country_indicators.csv")
    known = set(ind.iso3)
    dq = DQ("silver"); quarantine = []

    def q(df, reason, table):
        if len(df):
            d = df.copy(); d["_reason"] = reason; d["_table"] = table; quarantine.append(d.astype(str))

    # --- suppliers
    sup = pd.read_parquet(b / "suppliers_erp.parquet")
    n_raw = len(sup)
    sup["supplier_id"] = sup.supplier_id.str.strip().str.upper()
    dup = sup.duplicated("supplier_id", keep="first"); q(sup[dup], "duplicate_supplier_id", "suppliers"); sup = sup[~dup]
    sup["country_iso3"] = [to_iso3(c, known) for c in sup.country]
    q(sup[sup.country_iso3.isna()], "unmapped_country", "suppliers"); sup = sup[sup.country_iso3.notna()]
    sup["tier_reported"] = pd.to_numeric(sup.tier, errors="coerce")
    sup["annual_spend_usd"] = pd.to_numeric(sup.annual_spend_usd, errors="coerce")
    neg = sup.annual_spend_usd < 0; q(sup[neg], "negative_spend_nulled", "suppliers"); sup.loc[neg, "annual_spend_usd"] = np.nan
    sup = sup[["supplier_id", "name", "country_iso3", "sector", "tier_reported", "annual_spend_usd"]].reset_index(drop=True)
    ids = set(sup.supplier_id)

    # --- edges
    ed = pd.read_parquet(b / "supply_edges.parquet")
    ed["from_supplier"] = ed.from_supplier.str.strip().str.upper(); ed["to_supplier"] = ed.to_supplier.str.strip().str.upper()
    orphan = ~(ed.from_supplier.isin(ids) & ed.to_supplier.isin(ids)); q(ed[orphan], "orphan_edge", "edges"); ed = ed[~orphan]
    loop = ed.from_supplier == ed.to_supplier; q(ed[loop], "self_loop", "edges"); ed = ed[~loop]
    d2 = ed.duplicated(["from_supplier", "to_supplier"]); q(ed[d2], "duplicate_edge", "edges"); ed = ed[~d2]
    ed["share"] = pd.to_numeric(ed.share_pct) / 100
    n_share_clipped = int((ed.share > 1).sum()); ed["share"] = ed.share.clip(0, 1)   # noisy sole-source rows can exceed 100%
    ed["alt_known"] = ed.alt_source.isin(["Y", "N"])
    ed["alt_source"] = ed.alt_source.eq("Y")
    ed["valid_from_month"] = pd.PeriodIndex(ed.valid_from_month, freq="M")
    ed = ed[["from_supplier", "to_supplier", "share", "alt_source", "alt_known", "valid_from_month"]].reset_index(drop=True)

    derived = _derive_tier(sup.supplier_id, ed)
    sup["tier_derived"] = sup.supplier_id.map(derived).astype(int)
    sup["tier"] = sup.tier_reported.fillna(sup.tier_derived).astype(int)

    # --- signals
    sg = pd.read_parquet(b / "signals_monthly.parquet")
    sg["supplier_id"] = sg.supplier_id.str.strip().str.upper()
    o = ~sg.supplier_id.isin(ids); q(sg[o], "orphan_signal", "signals"); sg = sg[~o]
    for c in ["otd", "fin_score", "news_neg", "price_vol"]:
        sg[c] = pd.to_numeric(sg[c], errors="coerce")
    pct = sg.otd > 1.0; n_pct = int(pct.sum()); sg.loc[pct, "otd"] = sg.loc[pct, "otd"] / 100
    sg["otd"] = sg.otd.clip(0, 1)
    sg["period"] = pd.PeriodIndex(sg.period, freq="M"); sg["ingested_at"] = pd.to_datetime(sg.ingested_at)
    sg = sg.drop_duplicates(["supplier_id", "period", "ingested_at", "otd", "fin_score", "news_neg", "price_vol"])
    # a record becomes usable at the first month-end on/after its ingestion date
    sg["avail_period"] = sg.ingested_at.dt.to_period("M")   # usable by an as-of date at the end of this month
    sg = sg.sort_values(["supplier_id", "period", "ingested_at"]).reset_index(drop=True)
    sg["version"] = sg.groupby(["supplier_id", "period"]).cumcount() + 1
    sg = sg[["supplier_id", "period", "version", "otd", "fin_score", "news_neg", "price_vol", "ingested_at", "avail_period"]]

    # --- incidents / bom
    inc = pd.read_parquet(b / "incidents.parquet")
    inc["supplier_id"] = inc.supplier_id.str.strip().str.upper()
    inc["start_date"] = pd.to_datetime(inc.start_date); inc["end_date"] = pd.to_datetime(inc.end_date)
    inc["severity"] = pd.to_numeric(inc.severity)
    bad = ~inc.supplier_id.isin(ids) | (inc.end_date < inc.start_date); q(inc[bad], "bad_incident", "incidents"); inc = inc[~bad]
    inc["start_period"] = inc.start_date.dt.to_period("M")
    inc = inc[["supplier_id", "start_date", "end_date", "start_period", "category", "severity"]].reset_index(drop=True)
    bom = pd.read_parquet(b / "products_bom.parquet")
    bom["supplier_id"] = bom.supplier_id.str.strip().str.upper()
    bom["bom_share"] = pd.to_numeric(bom.bom_share); bom["annual_revenue_usd"] = pd.to_numeric(bom.annual_revenue_usd)
    bom = bom[["product", "supplier_id", "bom_share", "annual_revenue_usd"]]

    # ---------------- data-quality checks (after cleaning) ----------------
    dq.check("suppliers_unique_id", sup.supplier_id.is_unique, int(sup.supplier_id.duplicated().sum()), "== 0 duplicates")
    dq.check("suppliers_country_mapped", sup.country_iso3.isin(known).all(), int((~sup.country_iso3.isin(known)).sum()), "all ISO3 in reference")
    dq.check("suppliers_spend_positive", (sup.annual_spend_usd.dropna() > 0).all(), int((sup.annual_spend_usd < 0).sum()), "no negatives")
    dq.fraction_le("suppliers_spend_null_rate", sup.annual_spend_usd.isna().sum(), len(sup), 0.05, "warn")
    agree = (sup.tier_reported.dropna() == sup.tier_derived[sup.tier_reported.notna()]).mean()
    dq.check("tier_reported_vs_graph_derived_agreement", agree >= 0.85, round(float(agree), 3), ">= 0.85 (skip-tier edges explain the rest)", "warn")
    dq.check("edges_no_self_loops", (ed.from_supplier != ed.to_supplier).all(), 0, "== 0")
    dq.check("edges_endpoints_known", ed.from_supplier.isin(ids).all() and ed.to_supplier.isin(ids).all(), 0, "referential integrity")
    sums = ed.groupby("to_supplier").share.sum()
    dq.fraction_le("edges_customer_share_sum_off", int(((sums - 1).abs() > 0.05).sum()), len(sums), 0.02, "warn")
    dq.check("edges_share_range", ed.share.between(0, 1).all(), float(ed.share.max()), "within [0,1]")
    dq.fraction_le("edges_alt_unknown_rate", int((~ed.alt_known).sum()), len(ed), 0.10, "warn")
    dq.check("signals_otd_range", sg.otd.dropna().between(0, 1).all(), float(sg.otd.max()), "within [0,1] after unit fix")
    dq.check("signals_fin_range", sg.fin_score.dropna().between(-50, 150).all(), float(sg.fin_score.max()), "plausible range", "warn")
    dq.check("signals_news_nonneg", (sg.news_neg.dropna() >= 0).all(), float(sg.news_neg.min()), ">= 0")
    dq.check("signals_avail_after_period", (sg.avail_period >= sg.period).all(), 0, "ingestion never precedes the period (point-in-time safety)")
    for c in ["otd", "fin_score", "news_neg", "price_vol"]:
        dq.fraction_le(f"signals_{c}_null_rate", sg[c].isna().sum(), len(sg), 0.75, "warn")
    cov = sg.groupby("period").supplier_id.nunique() / len(sup)
    dq.check("signals_monthly_coverage_min", cov.min() >= 0.9, round(float(cov.min()), 3), ">= 0.9 of suppliers have a record each month")
    dq.check("incidents_have_supplier", inc.supplier_id.isin(ids).all(), 0, "referential integrity")
    dq.check("incidents_dates_ordered", (inc.end_date >= inc.start_date).all(), 0, "end >= start")
    dq.check("bom_share_sums_to_one", bool((bom.groupby("product").bom_share.sum().sub(1).abs() < 0.01).all()), 0, "per product")
    dq.check("bom_suppliers_known", bom.supplier_id.isin(ids).all(), 0, "referential integrity")
    # graph acyclicity (propagation assumes finite depth; cycles would be fine for the solver but are suspicious here)
    dq.check("graph_depth_finite", int(derived.max()) <= 6, int(derived.max()), "derived max tier <= 6", "warn")
    dq.raise_if_failed()

    sup.to_parquet(s / "suppliers.parquet", index=False); ed.to_parquet(s / "edges.parquet", index=False)
    sg.to_parquet(s / "signals.parquet", index=False); inc.to_parquet(s / "incidents.parquet", index=False)
    bom.to_parquet(s / "bom.parquet", index=False); ind.to_parquet(s / "country_indicators.parquet", index=False)
    qt = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=["_reason", "_table"])
    qt.to_parquet(s / "_quarantine.parquet", index=False)
    summary = qt.groupby(["_table", "_reason"]).size().rename("rows").reset_index().to_dict("records") if len(qt) else []
    dq.write(config.reports_root() / "dq_silver.json")
    info = dict(quarantine=summary, edge_share_clipped=n_share_clipped, otd_unit_fixed_rows=n_pct, raw_supplier_rows=n_raw, clean_suppliers=len(sup),
                edges=len(ed), signal_versions=len(sg))
    (s / "_summary.json").write_text(json.dumps(info, indent=2, default=str))
    return info
