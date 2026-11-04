import pytest
from fastapi.testclient import TestClient

from srg import score, api


@pytest.fixture(scope="module")
def client(world):
    score.write_serving_artifacts(world["silver"], world["g"])
    api.reset_cache()
    return TestClient(api.app)


def test_health_and_about(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["suppliers"] == 400 and r["edges"] > 400
    assert "simulated" in client.get("/about").json()["data"]


def test_suppliers_ranked_and_filtered(client):
    rows = client.get("/suppliers?limit=20").json()
    p = [r["p_disruption_3m"] for r in rows]
    assert p == sorted(p, reverse=True) and all(0 <= x <= 1 for x in p)
    t1 = client.get("/suppliers?tier=1&limit=500").json()
    assert t1 and all(r["tier"] == 1 for r in t1)


def test_supplier_detail_and_404(client):
    sid = client.get("/suppliers?limit=1").json()[0]["supplier_id"]
    d = client.get(f"/suppliers/{sid.lower()}").json()
    assert d["supplier_id"] == sid and isinstance(d["top_upstream_drivers"], list) and "suppliers" in d
    assert client.get("/suppliers/NOPE").status_code == 404


def test_subgraph(client):
    sid = client.get("/suppliers?tier=1&limit=1").json()[0]["supplier_id"]
    g = client.get(f"/graph/{sid}?depth=2").json()
    ids = {n["supplier_id"] for n in g["nodes"]}
    assert sid in ids and all(e["from_supplier"] in ids and e["to_supplier"] in ids for e in g["edges"])


def test_what_if_propagates_downstream_only(client):
    edges = api.store()["edges"]
    sup = edges.from_supplier.iloc[0]; cust = edges.to_supplier.iloc[0]
    r = client.post("/simulate", json={"shocked": [sup]}).json()
    impacted = {x["supplier_id"]: x["impact"] for x in r["impacted"]}
    assert cust in impacted and 0 < impacted[cust] <= 1 and sup not in impacted
    assert client.post("/simulate", json={"shocked": ["NOPE"]}).status_code == 404
    assert client.post("/simulate", json={"shocked": []}).status_code == 422


def test_products_and_dashboard(client):
    assert client.get("/products").json()[0]["expected_revenue_exposed_usd"] >= 0
    html = client.get("/").text
    assert "Simulated data" in html
