"""Fetch the real public country-level data used as simulator priors.

Sources (all public, no API key):
  * World Bank Worldwide Governance Indicators (political stability, govt effectiveness, rule of law)
  * World Bank Logistics Performance Index (overall score)
  * OFAC SDN list, address file: number of sanctioned-party addresses per country

Writes data/external/country_indicators.csv and data/external/SOURCES.json (retrieval time + raw row counts).
The committed snapshot is what the pipeline uses; re-run this script only to refresh it.
"""
import csv, io, json, datetime as dt, sys
from pathlib import Path
from urllib.request import urlopen, Request
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "external"
WB = "https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=2018:2024"
INDICATORS = {"GOV_WGI_PV.EST": "political_stability", "GOV_WGI_GE.EST": "gov_effectiveness",
              "GOV_WGI_RL.EST": "rule_of_law", "LP.LPI.OVRL.XQ": "lpi_overall"}
OFAC_ADD = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ADD.CSV"
# OFAC address file uses country *names*; map to ISO3 for the countries we simulate.
NAME2ISO = {"China": "CHN", "Vietnam": "VNM", "India": "IND", "Mexico": "MEX", "Germany": "DEU", "Taiwan": "TWN",
            "United States": "USA", "Turkey": "TUR", "Russia": "RUS", "Ukraine": "UKR", "Brazil": "BRA",
            "Indonesia": "IDN", "Thailand": "THA", "Malaysia": "MYS", "Poland": "POL", "Japan": "JPN",
            "Korea, South": "KOR", "Bangladesh": "BGD", "Pakistan": "PAK", "Egypt": "EGY", "South Africa": "ZAF",
            "Nigeria": "NGA", "United Arab Emirates": "ARE", "Netherlands": "NLD", "Italy": "ITA",
            "United Kingdom": "GBR", "France": "FRA", "Canada": "CAN", "Philippines": "PHL", "Iran": "IRN",
            "Syria": "SYR", "Venezuela": "VEN", "Cuba": "CUB", "Burma": "MMR", "Hong Kong": "HKG"}


def get(url):
    with urlopen(Request(url, headers={"User-Agent": "supplier-risk-graph-fetch"}), timeout=60) as r:
        return r.read()


def wb(ind):
    rows = json.loads(get(WB.format(ind=ind)))[1]
    df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
                       for r in rows if r["value"] is not None and r["countryiso3code"]])
    return df.sort_values("year").groupby("iso3").tail(1).set_index("iso3").value.rename(INDICATORS[ind])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = [wb(i) for i in INDICATORS]
    df = pd.concat(cols, axis=1)
    raw = get(OFAC_ADD).decode("latin-1")
    counts = {}
    for row in csv.reader(io.StringIO(raw)):
        if len(row) >= 5:
            counts[row[4].strip()] = counts.get(row[4].strip(), 0) + 1
    df["ofac_sdn_addresses"] = pd.Series({NAME2ISO[n]: c for n, c in counts.items() if n in NAME2ISO})
    df["ofac_sdn_addresses"] = df["ofac_sdn_addresses"].fillna(0).astype(int)
    keep = sorted(set(NAME2ISO.values()))
    df = df.loc[df.index.intersection(keep)].round(4)
    df.index.name = "iso3"
    df.to_csv(OUT / "country_indicators.csv")
    (OUT / "SOURCES.json").write_text(json.dumps({
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "world_bank": WB.format(ind="<indicator>"), "indicators": INDICATORS,
        "ofac_file": OFAC_ADD, "ofac_rows": sum(counts.values()), "countries_kept": len(df)}, indent=2))
    print(df)


if __name__ == "__main__":
    sys.exit(main())
