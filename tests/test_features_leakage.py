import numpy as np
import pandas as pd

from srg import config, evaluate as E, features as F, leakage as L, models as M


def test_labels_match_incident_log(world):
    g, s = world["g"], world["silver"]
    p0 = s["signals"].period.min()
    inc = s["incidents"].assign(t=lambda d: (d.start_period.astype("int64") - p0.ordinal)).groupby("supplier_id").t.apply(set).to_dict()
    lab = g[g.y.notna()].sample(3000, random_state=0)
    expect = [float(any(t0 + 1 <= x <= t0 + config.HORIZON for x in inc.get(sid, ()))) for sid, t0 in zip(lab.supplier_id, lab.t)]
    assert np.array_equal(lab.y.to_numpy(), np.array(expect))
    last = g.t.max()
    assert g[g.t > last - config.HORIZON].y.isna().all() and g[g.t <= last - config.HORIZON].y.notna().all()


def test_prevalence_is_low_and_nontrivial(world):
    y = world["g"].y.dropna()
    assert 0.005 < y.mean() < 0.12


def test_point_in_time_truncation(world):
    r = L.check_point_in_time(world["silver"], world["g"], t0s=(30,))
    assert r["passed"], r["detail"]


def test_features_use_first_seen_not_final_value(world):
    """A late correction (version>=2) must never change a past feature row: compare with a build that drops v2+."""
    s = dict(world["silver"]); s["signals"] = s["signals"][s["signals"].version == 1]
    g1 = F.build_features(s)
    cols = F.OWN_FEATURES
    a = world["g"].sort_values(["t", "supplier_id"])[cols].to_numpy(float); b = g1.sort_values(["t", "supplier_id"])[cols].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_no_latents_and_purge(world):
    assert L.check_no_latents(world["silver"], world["g"])["passed"]
    assert L.check_purge_gap()["passed"]


def test_canary_leak_is_detected(world):
    r = L.check_canary(world["g"])
    assert r["passed"], r["detail"]


def test_label_shuffle_runs(world):
    r = L.check_label_shuffle(world["g"], n=2)
    assert "label_shuffle_null" == r["name"] and "ROC-AUC" in r["detail"]


def test_metrics_helpers():
    df = pd.DataFrame(dict(t=[0] * 10, y=[0] * 8 + [1, 1]))
    perfect = np.r_[np.zeros(8), np.ones(2)]
    m = E.metrics(df, perfect, calibrated=False)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["recall_at_5pct"] == 0.5
    assert E.ece(np.array([0, 1] * 50), np.r_[np.linspace(0.45, 0.46, 50), np.linspace(0.54, 0.55, 50)]) > 0  # runs on near-constant scores
    assert abs(E.ece(np.array([0] * 90 + [1] * 10), np.r_[np.full(50, 0.1), np.full(50, 0.1)]) ) < 0.05


def test_feature_groups_disjoint():
    groups = [F.OWN_FEATURES, F.STATIC_FEATURES, F.GRAPH_STRUCT, F.GRAPH_CONTAGION, F.PLACEBO]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat))


def test_evaluation_smoke(world):
    ev = E.run_evaluation(world["g"], n_boot=5, quick=True)
    assert ev["splits"]["test"]["positives"] > 0
    assert ev["models"]["base_rate"]["roc_auc"] == 0.5
    assert 0 < ev["models"]["hgb_full"]["roc_auc"] <= 1
    assert set(ev["ablations"]) >= {"minus_graph_contagion", "plus_placebo_downstream", "full"}
