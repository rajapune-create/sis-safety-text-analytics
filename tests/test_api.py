from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_analyse_links_and_classifies():
    r = client.post("/analyse", json={"text": "SIF-20-014 tripped on pressure high (U20-PT-2014 reached trip "
                                              "setpoint), reactor R-2101 shut down safely. Power dip on substation "
                                              "bus, several motors stopped.", "unit": "U20"}).json()
    assert r["sif_id"] == "SIF-20-014" and r["link_method"] == "sif_id"
    assert r["outcome"] == "demand"
    assert 0 <= r["confidence"] <= 1


def test_unlinked_narrative_is_sent_for_review():
    r = client.post("/analyse", json={"text": "unit upset, cause under investigation", "unit": "U99"}).json()
    assert r["sif_id"] is None and r["needs_review"] is True
