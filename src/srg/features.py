"""Gold layer: point-in-time feature panel (supplier x month) with labels.

POINT-IN-TIME RULE: a feature row for as-of month t may only use
  * signal records whose period <= t AND whose ingestion landed in a month <= t  (first-seen version only - later
    corrections are ignored, exactly as a live system would see them),
  * incidents that STARTED in a month <= t,
  * graph edges with valid_from <= t.
`tests/test_leakage.py` proves this by deleting all later data and recomputing.
Label y(t) = 1 if the supplier has a NEW incident starting in months t+1 .. t+H.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .graph import build_w, noisy_or_propagate, pass_weights
from .simulate import country_risk

SIGNALS = ["otd", "fin_score", "news_neg", "price_vol"]
STATIC_IND = ["political_stability", "gov_effectiveness", "rule_of_law", "lpi_overall"]


def _own_composite(otd3, fin, news3):
    """Fixed (not fitted) 0-1 distress score from observed signals; NaN-aware weighted average of three evidence terms."""
    terms = [(0.5, np.clip((0.93 - otd3) / 0.15, 0, 1)), (0.3, np.clip(news3 / 3, 0, 1)), (0.2, np.clip((55 - fin) / 40, 0, 1))]
    num = sum(w * np.nan_to_num(v) for w, v in terms)
    den = sum(w * ~np.isnan(v) for w, v in terms)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def build_features(silver: dict, horizon: int = config.HORIZON, max_t: int | None = None, n_months: int | None = None):
    sup = silver["suppliers"].sort_values("supplier_id").reset_index(drop=True)
    ids = sup.supplier_id.tolist(); idx = {s: i for i, s in enumerate(ids)}; N = len(ids)
    sg, ed, inc, ind = silver["signals"], silver["edges"], silver["incidents"], silver["country_indicators"]
    p0 = sg.period.min(); T = int(sg.period.max().ordinal - p0.ordinal) + 1
    if n_months is not None:
        T = n_months
    if max_t is not None:
        T = min(T, max_t + 1)
    m_of = lambda p: (pd.PeriodIndex(p).astype("int64") - p0.ordinal).to_numpy()

    # first-seen version of each (supplier, period), with the month it became usable
    first = sg[sg.version == 1].copy()
    first["m"] = m_of(first.period); first["a"] = m_of(first.avail_period); first["i"] = first.supplier_id.map(idx)
    first = first[(first.m < T)]
    avail = np.full((T, N), 10**6); avail[first.m.to_numpy(), first.i.to_numpy()] = first.a.to_numpy()
    mats = {}
    for c in SIGNALS:
        M = np.full((T, N), np.nan); M[first.m.to_numpy(), first.i.to_numpy()] = first[c].to_numpy(); mats[c] = M

    # incidents by start month
    inc = inc.copy(); inc["m"] = m_of(inc.start_period); inc["i"] = inc.supplier_id.map(idx)
    starts = np.zeros((T + horizon + 1, N), bool)                      # labels may look beyond T only if data exist
    ok = inc.m < starts.shape[0]
    starts[inc.m[ok].to_numpy(), inc.i[ok].to_numpy()] = True
    last_m = int(sg.period.max().ordinal - p0.ordinal)               # last month with data in the full dataset
    sev = np.zeros_like(starts, float); sev[inc.m[ok].to_numpy(), inc.i[ok].to_numpy()] = inc.severity[ok].to_numpy()

    # static
    crisk = country_risk(ind.set_index("iso3")); cr = crisk.reindex(sup.country_iso3).to_numpy()
    indc = ind.set_index("iso3").reindex(sup.country_iso3)
    sector_code = sup.sector.astype("category").cat.codes.to_numpy()
    static = pd.DataFrame(dict(tier=sup.tier.to_numpy(), country_risk=cr, sector_code=sector_code,
                               log_spend=np.log1p(sup.annual_spend_usd.fillna(sup.annual_spend_usd.median())).to_numpy(),
                               spend_missing=sup.annual_spend_usd.isna().to_numpy().astype(int),
                               ofac_log=np.log1p(indc.ofac_sdn_addresses.to_numpy())))
    for c in STATIC_IND:
        static[c] = indc[c].to_numpy()

    # edges
    e = ed.copy(); e["c"] = e.to_supplier.map(idx); e["s"] = e.from_supplier.map(idx)
    e["vf"] = m_of(e.valid_from_month); e["w"] = pass_weights(e.share, e.alt_source)
    c_all, s_all, w_all, sh_all, vf_all, alt_all = (e[k].to_numpy() for k in ["c", "s", "w", "share", "vf", "alt_source"])

    rows = []
    for t in range(config.WARMUP, T):
        vis = {c: np.where((avail[: t + 1] <= t), mats[c][: t + 1], np.nan) for c in SIGNALS}
        F = {}
        otd = pd.DataFrame(vis["otd"]); fin = pd.DataFrame(vis["fin_score"]); news = pd.DataFrame(vis["news_neg"]); pv = pd.DataFrame(vis["price_vol"])
        otd_last = otd.ffill(limit=2).iloc[t].to_numpy()
        otd3 = otd.iloc[t - 2:].mean().to_numpy(); otd6 = otd.iloc[t - 5:].mean().to_numpy()
        otd_prev = otd.iloc[max(t - 8, 0): t - 2].mean().to_numpy()
        F.update(otd_last=otd_last, otd_mean3=otd3, otd_mean6=otd6, otd_trend=otd3 - otd_prev,
                 otd_min6=otd.iloc[t - 5:].min().to_numpy(), otd_missing6=otd.iloc[t - 5:].isna().mean().to_numpy())
        fin_last = fin.ffill(limit=5).iloc[t].to_numpy()
        F.update(fin_last=fin_last, fin_missing=np.isnan(fin_last).astype(float))
        n3 = news.iloc[t - 2:].sum(min_count=1).to_numpy(); n6 = news.iloc[t - 5:].sum(min_count=1).to_numpy()
        F.update(news_sum3=n3, news_sum6=n6, news_missing6=news.iloc[t - 5:].isna().mean().to_numpy())
        F.update(price_vol3=pv.iloc[t - 2:].mean().to_numpy())
        # incidents (start <= t)
        past = starts[: t + 1]
        n12 = past[max(0, t - 11): t + 1].sum(0)
        since = np.minimum(np.where(past.any(0), np.argmax(past[::-1], axis=0), 36), 36).astype(float)
        F.update(inc_12m=n12.astype(float), inc_total=past.sum(0).astype(float), months_since_inc=since,
                 last_severity=np.array([sev[:t + 1][np.where(past[:, i])[0][-1], i] if past[:, i].any() else 0 for i in range(N)]))
        own = _own_composite(otd3, fin_last, np.where(np.isnan(n3), np.nan, n3 / 3))
        F["own_composite"] = own

        # graph (edges valid at t)
        v = vf_all <= t
        c, s, w, sh = c_all[v], s_all[v], w_all[v], sh_all[v]
        W = build_w(N, c, s, w); S = build_w(N, c, s, sh)
        s0 = np.nan_to_num(own, nan=0.0)
        has_up = np.asarray((W > 0).sum(1)).ravel()
        wsum = np.asarray(W.sum(1)).ravel()
        up1 = W @ s0
        up1_mean = np.where(wsum > 0, up1 / np.maximum(wsum, 1e-9), 0)
        up1_max = np.zeros(N); np.maximum.at(up1_max, c, w * s0[s])
        logq = np.zeros(N); np.add.at(logq, c, np.log1p(-np.minimum(w * s0[s], 0.999999)))
        up_or = 1 - np.exp(logq)
        up2 = W @ (W @ s0)
        prev_own = _own_composite(otd.iloc[t - 5:t - 2].mean().to_numpy(), fin.ffill(limit=5).iloc[t - 3].to_numpy(),
                                  news.iloc[t - 5:t - 2].sum(min_count=1).to_numpy() / 3)
        up_trend = W @ (s0 - np.nan_to_num(prev_own))
        prop = noisy_or_propagate(s0, c, s, w)
        hhi = np.zeros(N); np.add.at(hhi, c, sh ** 2)
        crit = np.ones(N)
        for _ in range(6):
            crit = 1 + np.asarray(S.T @ crit).ravel()    # value-at-stake: own weight plus dependency-weighted customers'
        n_cust = np.bincount(s, minlength=N)
        cust_mean = np.where(n_cust > 0, np.asarray(sp.csr_matrix((np.ones(len(s)), (s, c)), shape=(N, N)) @ s0).ravel() / np.maximum(n_cust, 1), 0)
        up_cr = np.where(wsum > 0, (W @ np.nan_to_num(cr)) / np.maximum(wsum, 1e-9), 0)
        up_hi = np.where(has_up > 0, np.asarray(sp.csr_matrix(((cr[s] > 1).astype(float), (c, s)), shape=(N, N)).sum(1)).ravel() / np.maximum(has_up, 1), 0)
        alt_frac = np.zeros(N); np.add.at(alt_frac, c, sh * alt_all[v])
        F.update(g_n_suppliers=has_up.astype(float), g_sole_source=(has_up == 1).astype(float), g_hhi=hhi, g_alt_share=alt_frac,
                 g_n_customers=n_cust.astype(float), g_criticality=crit, g_up_country_risk=up_cr, g_up_hi_risk_share=up_hi,
                 g_up1_mean=up1_mean, g_up1_max=up1_max, g_up1_or=up_or, g_up2=up2, g_up_trend=up_trend,
                 g_prop_risk=prop, g_prop_gain=prop - s0, g_down_mean=cust_mean)

        df = pd.DataFrame(F); df.insert(0, "supplier_id", ids); df.insert(1, "t", t)
        df = pd.concat([df, static], axis=1)
        fut = starts[t + 1: t + 1 + horizon]
        known_until = last_m                                         # label is observable only if t+H <= last month
        df["y"] = fut.any(0).astype(float) if t + horizon <= known_until else np.nan
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["month"] = (p0 + out.t).astype(str)
    return out


OWN_FEATURES = ["otd_last", "otd_mean3", "otd_mean6", "otd_trend", "otd_min6", "otd_missing6", "fin_last", "fin_missing",
                "news_sum3", "news_sum6", "news_missing6", "price_vol3", "inc_12m", "inc_total", "months_since_inc",
                "last_severity", "own_composite"]
STATIC_FEATURES = ["tier", "country_risk", "sector_code", "log_spend", "spend_missing", "ofac_log"] + STATIC_IND
GRAPH_STRUCT = ["g_n_suppliers", "g_sole_source", "g_hhi", "g_alt_share", "g_n_customers", "g_criticality",
                "g_up_country_risk", "g_up_hi_risk_share"]
GRAPH_CONTAGION = ["g_up1_mean", "g_up1_max", "g_up1_or", "g_up2", "g_up_trend", "g_prop_risk", "g_prop_gain"]
PLACEBO = ["g_down_mean"]
ALL_FEATURES = OWN_FEATURES + STATIC_FEATURES + GRAPH_STRUCT + GRAPH_CONTAGION
