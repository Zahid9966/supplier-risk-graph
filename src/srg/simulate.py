"""Hard-ish multi-tier supply-network simulator.

WHAT IS REAL: the country covariates (World Bank WGI/LPI, OFAC SDN address counts) in data/external.
WHAT IS SYNTHETIC: everything else - suppliers, edges, monthly signals, disruption incidents.

Generative story (monthly, 60 months):
  hazard_i(t) = sigmoid( b0 + b_c*country_risk_i + b_s*sector_i + b_u*u_i      # static, partly UNOBSERVED (u_i)
                         + b_f*fin_stress_i(t)                                  # latent AR(1), observed only noisily
                         + gamma * P_i(t)                                       # CONTAGION from disrupted upstream
                         + b_x*country_shock_c(t) )                             # common regional shocks
  P_i(t) = 1 - prod_j (1 - pass_ij * a_j),  a_j = max(act_j(t-1), .7 act_j(t-2), .4 act_j(t-3)),  pass_ij = dependency_share_ij * (0.5 if alt source else 1)
Disruptions last 1-6 months. `gamma` is the contagion strength; gamma=0 gives the no-contagion control world.

Observed signals are noisy, partially missing, quarterly-stale (financials) and delivered with ingestion lag and
corrections. Ground truth (latents, causes) goes to data/sim_truth and is NEVER read by the pipeline except by the
explicitly labelled oracle baseline and tests.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from . import config

SECTORS = ["electronics", "metals", "chemicals", "plastics", "logistics", "packaging"]
SECTOR_EFFECT = dict(zip(SECTORS, [0.35, 0.1, 0.3, -0.1, 0.4, -0.3]))
# rough sourcing weights (synthetic choice, not trade data)
COUNTRY_W = {"CHN": 18, "IND": 9, "VNM": 7, "MEX": 8, "DEU": 8, "USA": 10, "TUR": 5, "KOR": 4, "JPN": 4, "POL": 3,
             "THA": 3, "MYS": 3, "IDN": 3, "BRA": 3, "BGD": 2, "PAK": 2, "EGY": 1.5, "ZAF": 1.5, "NGA": 1,
             "UKR": 1.5, "RUS": 2, "ITA": 3, "FRA": 2, "GBR": 2, "NLD": 2, "MMR": 1, "PHL": 2, "ARE": 2}
COUNTRY_VARIANTS = {"CHN": ["CN", "China", "china", "PRC"], "VNM": ["VN", "Viet Nam", "Vietnam"],
                    "USA": ["US", "United States", "U.S.A."], "DEU": ["DE", "Germany", "Deutschland"],
                    "IND": ["IN", "India"], "MEX": ["MX", "Mexico", "México"], "TUR": ["TR", "Turkey", "Türkiye"],
                    "KOR": ["KR", "South Korea", "Korea, Rep."], "JPN": ["JP", "Japan"], "GBR": ["GB", "UK"]}
INCIDENT_TYPES = ["financial_distress", "facility_damage", "labor_dispute", "cyber_incident", "logistics_failure"]


@dataclass
class SimConfig:
    seed: int = 42
    n_months: int = 60
    tier_sizes: tuple = (100, 300, 700, 900)
    gamma: float = 7.0          # contagion strength (0 = control world)
    base_logit: float = -5.7
    start: str = "2019-01"
    inject_defects: bool = True


def country_risk(ind: pd.DataFrame) -> pd.Series:
    """Composite country risk (z-scored mean of governance/logistics/sanctions indicators). Higher = riskier."""
    z = lambda s: (s - s.mean()) / s.std()
    comp = (-z(ind.political_stability) - z(ind.gov_effectiveness) - z(ind.rule_of_law)
            - z(ind.lpi_overall) + 0.5 * z(np.log1p(ind.ofac_sdn_addresses))) / 4.5
    return (comp - comp.mean()) / comp.std()


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(cfg: SimConfig, out_raw, out_truth):
    rng = np.random.default_rng(cfg.seed)
    ind = pd.read_csv(config.EXTERNAL / "country_indicators.csv").set_index("iso3")
    crisk = country_risk(ind)
    iso = [c for c in COUNTRY_W if c in crisk.index]
    months = pd.period_range(cfg.start, periods=cfg.n_months, freq="M")
    T = cfg.n_months

    # ---------------- suppliers ----------------
    tiers = np.concatenate([[k + 1] * n for k, n in enumerate(cfg.tier_sizes)])
    N = len(tiers)
    ids = np.array([f"SUP{i:04d}" for i in range(N)])
    country = np.empty(N, dtype=object)
    for k in range(1, 5):  # deeper tiers skew toward riskier countries
        idx = np.where(tiers == k)[0]
        w = np.array([COUNTRY_W[c] * np.exp(0.35 * (k - 1) * crisk[c]) for c in iso])
        country[idx] = rng.choice(iso, size=len(idx), p=w / w.sum())
    sector = rng.choice(SECTORS, size=N)
    size = rng.lognormal(0, 0.8, N) * (1.8 - 0.3 * tiers)       # relative size
    u = rng.normal(0, 1, N)                                       # UNOBSERVED supplier-specific fragility
    cr = crisk.reindex(country).to_numpy()
    sec = np.array([SECTOR_EFFECT[s] for s in sector])
    spend = np.round(rng.lognormal(13.2, 0.9, N) * (1.6 - 0.25 * tiers), 0)
    names = [f"{rng.choice(['Apex','Nova','Delta','Orion','Titan','Zenith','Vertex','Helix','Kora','Sable'])} "
             f"{s.title()} {i:04d}" for i, s in enumerate(sector)]

    # ---------------- edges (supplier -> customer), preferential attachment, some skip-tier ----------------
    pop = rng.pareto(1.6, N) + 1                                   # hub-ness
    edges = []
    for k in (2, 3, 4):
        sup_idx = np.where(tiers == k)[0]
        for j in sup_idx:
            n_cust = rng.choice([1, 2, 3], p=[0.55, 0.32, 0.13])
            tgt_tier = k - 1 if rng.random() > 0.08 else max(1, k - 2)
            cand = np.where(tiers == tgt_tier)[0]
            p = pop[cand] / pop[cand].sum()
            for c in rng.choice(cand, size=min(n_cust, len(cand)), replace=False, p=p):
                edges.append((j, int(c)))
    e = pd.DataFrame(edges, columns=["sup", "cust"]).drop_duplicates()
    # every tier<4 customer needs >=1 supplier
    for c in range(N):
        if tiers[c] < 4 and not (e.cust == c).any():
            j = rng.choice(np.where(tiers == tiers[c] + 1)[0])
            e.loc[len(e)] = (j, c)
    e["raw"] = rng.uniform(0.2, 1.0, len(e))
    e["share"] = e.raw / e.groupby("cust").raw.transform("sum")    # customer's dependency on supplier
    e["alt"] = rng.random(len(e)) < 0.4
    e["valid_from"] = np.where(rng.random(len(e)) < 0.12, rng.integers(6, T - 12, len(e)), 0)
    # edges that start later: renormalise is NOT done (share is share at the time it exists, documented)
    pass_ = (e.share * np.where(e.alt, 0.5, 1.0)).to_numpy()
    sj, cj, vf = e.sup.to_numpy(), e.cust.to_numpy(), e.valid_from.to_numpy()

    # ---------------- dynamics ----------------
    phi = 0.93
    fin = np.zeros((T, N)); fin[0] = rng.normal(0, 0.5, N) + 0.25 * cr + 0.3 * u
    for t in range(1, T):
        fin[t] = phi * fin[t - 1] + rng.normal(0, 0.28, N) + (1 - phi) * (0.25 * cr + 0.3 * u)
    ciso = np.array(iso); cshock = np.zeros((T, len(iso)))
    for t in range(T):
        hit = rng.random(len(iso)) < 0.012 * (1 + np.maximum(crisk[iso].to_numpy(), -0.5))
        cshock[t] += hit
        if t > 0:
            cshock[t] = np.maximum(cshock[t], 0.6 * cshock[t - 1] * (rng.random(len(iso)) < 0.7))
    cidx = pd.Series(range(len(iso)), index=iso)[country].to_numpy()
    cs = cshock[:, cidx]                                            # T x N

    active = np.zeros((T, N), bool); remaining = np.zeros(N, int)
    starts = np.zeros((T, N), bool); cause_contagion = np.zeros((T, N), bool)
    Pm = np.zeros((T, N)); hz = np.zeros((T, N))
    for t in range(T):
        if t > 0:
            valid = vf <= t
            # upstream disruption drains buffers with memory: effective activity = max(a[t-1], .7 a[t-2], .4 a[t-3])
            a_eff = active[t - 1].astype(float)
            if t > 1: a_eff = np.maximum(a_eff, 0.7 * active[t - 2])
            if t > 2: a_eff = np.maximum(a_eff, 0.4 * active[t - 3])
            pw = pass_ * a_eff[sj] * valid
            logq = np.zeros(N)
            np.add.at(logq, cj, np.log1p(-np.minimum(pw, 0.999)))
            Pm[t] = 1 - np.exp(logq)
        logit = (cfg.base_logit + 0.55 * cr + 0.5 * sec + 0.6 * u + 0.9 * fin[t] + cfg.gamma * Pm[t] + 1.4 * cs[t])
        hz[t] = _sigmoid(logit)
        free = remaining == 0
        new = free & (rng.random(N) < hz[t])
        starts[t] = new
        own_only = _sigmoid(logit - cfg.gamma * Pm[t])
        cause_contagion[t] = new & (rng.random(N) * hz[t] > own_only)   # attribution proxy for diagnostics
        dur = np.minimum(1 + rng.poisson(1.5, N), 6)
        remaining = np.where(new, dur, np.maximum(remaining - 1, 0))
        active[t] = remaining > 0 if t == 0 else (remaining > 0)
        # month t counts as active for the start month through the final remaining month
    # active[t] above = still disrupted after start at t (includes start month)

    # ---------------- observed signals ----------------
    miss_p = np.array([0.08, 0.18, 0.30, 0.40])[tiers - 1]
    fin_cov = rng.random(N) < np.array([0.95, 0.7, 0.4, 0.25])[tiers - 1]
    news_cov = (0.4 + 0.6 * np.clip(size / size.max() * 4, 0, 1))
    otd = 0.96 - 0.05 * fin - 0.25 * active - 0.10 * Pm + rng.normal(0, 0.04, (T, N))
    otd = np.clip(otd, 0.2, 1.0)
    otd[rng.random((T, N)) < miss_p] = np.nan
    fin_score = 60 - 18 * fin + rng.normal(0, 14, (T, N))
    for t in range(T):                                              # quarterly reporting, stale in between
        if t % 3 != 0:
            fin_score[t] = fin_score[t - 1]
    fin_score[:, ~fin_cov] = np.nan
    lam = (0.15 + 1.2 * active + 0.25 * np.maximum(fin, 0) + 0.5 * cs) * news_cov
    news_neg = rng.poisson(np.minimum(lam, 8) + 0.1 * (rng.random((T, N)) < 0.03) * 5)  # incl. false positives
    news_neg = news_neg.astype(float); news_neg[:, rng.random(N) < 0.35] = np.nan   # uncovered suppliers
    price_vol = np.abs(rng.normal(0, 0.02 + 0.03 * np.maximum(fin, 0) + 0.03 * active))
    price_vol[rng.random((T, N)) < miss_p] = np.nan

    # ---------------- raw (messy) files ----------------
    out_raw.mkdir(parents=True, exist_ok=True); out_truth.mkdir(parents=True, exist_ok=True)
    defects = {}

    def cname(c):
        if cfg.inject_defects and c in COUNTRY_VARIANTS and rng.random() < 0.45:
            return rng.choice(COUNTRY_VARIANTS[c])
        return c if not cfg.inject_defects else ({"CHN": "CN"} .get(c, c) if rng.random() < 0.0 else c)

    sup = pd.DataFrame(dict(supplier_id=ids, name=names, country=[cname(c) for c in country], sector=sector,
                            tier=tiers.astype(float), annual_spend_usd=spend))
    if cfg.inject_defects:
        m = rng.random(N) < 0.06; sup.loc[m, "tier"] = np.nan; defects["suppliers_missing_tier"] = int(m.sum())
        m = rng.random(N) < 0.015; sup.loc[m, "annual_spend_usd"] *= -1; defects["suppliers_negative_spend"] = int(m.sum())
        m = rng.random(N) < 0.1; sup.loc[m, "supplier_id"] = sup.loc[m, "supplier_id"].str.lower() + " "
        defects["suppliers_dirty_id"] = int(m.sum())
        dup = sup.sample(5, random_state=1); sup = pd.concat([sup, dup]); defects["suppliers_duplicate_rows"] = 5
    sup.to_csv(out_raw / "suppliers_erp.csv", index=False)

    ed = pd.DataFrame(dict(from_supplier=ids[sj], to_supplier=ids[cj],
                           share_pct=np.round(e.share.to_numpy() * 100 * rng.normal(1, 0.02, len(e)), 2),
                           alt_source=np.where(e.alt, "Y", "N"), valid_from_month=[str(months[v]) for v in vf]))
    if cfg.inject_defects:
        ed.loc[rng.random(len(ed)) < 0.05, "alt_source"] = ""
        ed = pd.concat([ed, ed.sample(12, random_state=2),
                        pd.DataFrame([dict(from_supplier="SUP9999", to_supplier=ids[0], share_pct=10, alt_source="N",
                                           valid_from_month="2019-01")] * 3),
                        pd.DataFrame([dict(from_supplier=ids[5], to_supplier=ids[5], share_pct=5, alt_source="N",
                                           valid_from_month="2019-01")])])
        defects.update(edges_duplicates=12, edges_orphans=3, edges_self_loops=1)
    ed.to_csv(out_raw / "supply_edges.csv", index=False)

    lag_days = rng.integers(2, 40, (T, N))
    tt, ii = np.meshgrid(np.arange(T), np.arange(N), indexing="ij")
    anyobs = ~(np.isnan(otd) & np.isnan(fin_score) & np.isnan(news_neg) & np.isnan(price_vol))
    eom = np.array([m.end_time.normalize() for m in months], dtype="datetime64[ns]")
    o_arr = np.round(otd, 4)
    if cfg.inject_defects:                                           # unit defect: percent instead of fraction
        o_arr = np.where(rng.random((T, N)) < 0.04, np.round(otd * 100, 2), o_arr)
    sig = pd.DataFrame(dict(
        supplier_id=ids[ii[anyobs]], period=np.array([str(m) for m in months])[tt[anyobs]],
        otd=o_arr[anyobs], fin_score=np.round(fin_score[anyobs], 2),
        news_neg=news_neg[anyobs], price_vol=np.round(price_vol[anyobs], 4),
        ingested_at=pd.to_datetime(eom[tt[anyobs]] + lag_days[anyobs].astype("timedelta64[D]")).strftime("%Y-%m-%d")))
    if cfg.inject_defects:
        corr = sig.sample(frac=0.03, random_state=3).copy()             # late corrections (same key, later ingest)
        small = corr.otd.notna() & (corr.otd <= 1)
        corr.loc[small, "otd"] = (corr.loc[small, "otd"] + rng.normal(0, 0.02, int(small.sum()))).clip(0, 1).round(4)
        corr["ingested_at"] = (pd.to_datetime(corr.ingested_at) + pd.Timedelta(days=75)).dt.strftime("%Y-%m-%d")
        orph = sig.sample(40, random_state=4).copy(); orph["supplier_id"] = "SUPXXXX"
        sig = pd.concat([sig, corr, orph]).sample(frac=1, random_state=5)
        defects.update(signals_corrections=len(corr), signals_orphans=40, signals_pct_unit_rows=int((sig.otd > 1).sum()))
    sig.to_json(out_raw / "signals_monthly.jsonl", orient="records", lines=True)

    inc = []
    for t, i in zip(*np.where(starts)):
        d = 1
        while t + d < T and active[t + d, i] and not starts[t + d, i]:
            d += 1
        inc.append(dict(supplier_id=ids[i], start_date=str(months[t].start_time.date()),
                        end_date=str((months[min(t + d - 1, T - 1)].end_time.date())),
                        category=str(rng.choice(INCIDENT_TYPES, p=[.25, .15, .2, .15, .25])),
                        severity=int(rng.integers(1, 6))))
    pd.DataFrame(inc).to_csv(out_raw / "incidents.csv", index=False)

    bom = []
    t1 = np.where(tiers == 1)[0]
    for p in range(16):
        pick = rng.choice(t1, size=rng.integers(3, 7), replace=False); w = rng.dirichlet(np.ones(len(pick)))
        rev = float(np.round(rng.lognormal(16.5, 0.5), -3))
        bom += [dict(product=f"PRD{p+1:02d}", supplier_id=ids[s], bom_share=round(float(x), 4), annual_revenue_usd=rev)
                for s, x in zip(pick, w)]
    pd.DataFrame(bom).to_csv(out_raw / "products_bom.csv", index=False)

    # ---------------- ground truth (never used by pipeline except oracle baseline/tests) ----------------
    pd.DataFrame(dict(supplier_id=ids, true_u=u, true_country_risk=cr)).to_parquet(out_truth / "supplier_latents.parquet")
    long = pd.DataFrame(dict(month=np.repeat(np.arange(T), N), supplier_id=np.tile(ids, T),
                             active=active.ravel(), start=starts.ravel(), p_upstream=Pm.ravel(),
                             fin_stress=fin.ravel(), hazard=hz.ravel(), contagion_start=cause_contagion.ravel()))
    long.to_parquet(out_truth / "dynamics.parquet")
    (out_truth / "manifest.json").write_text(json.dumps(dict(config=asdict(cfg), injected_defects=defects,
        months=[str(m) for m in months], n_suppliers=N, n_edges=len(e), n_incidents=len(inc),
        monthly_start_rate=float(starts.mean()), contagion_share_of_starts=float(cause_contagion.sum() / max(starts.sum(), 1))), indent=2, default=str))
    return json.loads((out_truth / "manifest.json").read_text())
