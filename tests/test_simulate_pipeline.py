import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from srg import config, pipeline
from srg.dq import DQ
from srg.simulate import SimConfig, simulate, country_risk


def test_simulation_is_deterministic(tmp_path):
    cfg = SimConfig(seed=3, tier_sizes=(8, 16, 24, 30), n_months=24)
    simulate(cfg, tmp_path / "a", tmp_path / "ta"); simulate(cfg, tmp_path / "b", tmp_path / "tb")
    for f in ("suppliers_erp.csv", "supply_edges.csv", "signals_monthly.jsonl", "incidents.csv"):
        assert hashlib.md5((tmp_path / "a" / f).read_bytes()).hexdigest() == hashlib.md5((tmp_path / "b" / f).read_bytes()).hexdigest()


def test_country_risk_uses_real_indicators():
    ind = pd.read_csv(config.EXTERNAL / "country_indicators.csv").set_index("iso3")
    r = country_risk(ind)
    assert abs(r.mean()) < 1e-9 and abs(r.std() - 1) < 1e-9
    assert r["MMR"] > r["DEU"] and r["VEN"] > r["JPN"]          # sanity: fragile states riskier than stable ones
    assert (config.EXTERNAL / "SOURCES.json").exists()


def test_bronze_is_lossless(world):
    raw = config.layer("raw"); b = config.layer("bronze")
    assert len(pd.read_parquet(b / "suppliers_erp.parquet")) == len(pd.read_csv(raw / "suppliers_erp.csv"))
    assert len(pd.read_parquet(b / "signals_monthly.parquet")) == sum(1 for _ in open(raw / "signals_monthly.jsonl"))
    assert {"_source_file", "_ingest_run", "_row_hash"} <= set(pd.read_parquet(b / "incidents.parquet").columns)


def test_silver_quarantine_matches_injected_defects(world):
    d = world["manifest"]["injected_defects"]
    q = pd.read_parquet(config.layer("silver") / "_quarantine.parquet")
    n = lambda table, reason: int(((q._table == table) & (q._reason == reason)).sum())
    assert n("edges", "duplicate_edge") == d["edges_duplicates"]
    assert n("edges", "orphan_edge") == d["edges_orphans"]
    assert n("edges", "self_loop") == d["edges_self_loops"]
    assert n("signals", "orphan_signal") == d["signals_orphans"]
    assert n("suppliers", "duplicate_supplier_id") == d["suppliers_duplicate_rows"]
    assert world["info"]["otd_unit_fixed_rows"] == d["signals_pct_unit_rows"]


def test_silver_is_clean(world):
    s = world["silver"]
    assert s["suppliers"].supplier_id.is_unique and s["suppliers"].supplier_id.str.match(r"^SUP\d{4}$").all()
    assert s["suppliers"].country_iso3.str.len().eq(3).all()
    assert s["signals"].otd.dropna().between(0, 1).all()
    assert (s["signals"].avail_period >= s["signals"].period).all()
    assert s["edges"].share.between(0, 1).all()
    assert not ((s["edges"].from_supplier == s["edges"].to_supplier).any())
    assert set(s["edges"].from_supplier) <= set(s["suppliers"].supplier_id)
    assert s["signals"].groupby(["supplier_id", "period"]).version.max().max() >= 2    # corrections preserved as versions


def test_dq_reports_written_and_no_critical_failures(world):
    rep = json.loads((config.reports_root() / "dq_silver.json").read_text())
    assert len(rep) >= 20 and not [r for r in rep if r["severity"] == "critical" and not r["passed"]]


def test_dq_framework_fails_loudly_on_bad_data():
    dq = DQ("t"); dq.check("x", False, 3, "== 0"); dq.check("y", False, 1, "ok", "warn")
    assert len(dq.failed_critical) == 1
    with pytest.raises(RuntimeError, match="x=3"):
        dq.raise_if_failed()


def test_silver_rejects_corrupted_bronze(world, tmp_path, monkeypatch):
    """Corrupt the bronze signals so every OTD is impossible: the critical DQ gate must stop the pipeline."""
    b = config.layer("bronze") / "signals_monthly.parquet"; orig = pd.read_parquet(b)
    try:
        bad = orig.copy(); bad["news_neg"] = -5
        bad.to_parquet(b, index=False)
        with pytest.raises(RuntimeError, match="signals_news_nonneg"):
            pipeline.run_silver()
    finally:
        orig.to_parquet(b, index=False); pipeline.run_silver()


def test_graph_has_hubs_and_skip_tier_edges(world):
    s = world["silver"]; e = s["edges"]; t = s["suppliers"].set_index("supplier_id").tier_derived
    tier_gap = (e.from_supplier.map(t) - e.to_supplier.map(t))
    assert (tier_gap > 1).any() or (tier_gap == 1).mean() > 0.8
    out_deg = e.groupby("from_supplier").size()
    in_deg = e.groupby("to_supplier").size()
    assert in_deg.max() >= 4                                   # preferential attachment creates hub customers
    assert out_deg.max() <= 4
