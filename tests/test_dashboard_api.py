import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from dashboard.backend.config import DashboardConfig
from dashboard.backend.server import create_app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DASHBOARD_API_TESTS") != "1",
    reason="FastAPI TestClient smoke tests are opt-in in lightweight environments",
)


@pytest.fixture
def client():
    return TestClient(create_app(DashboardConfig(mock=True)))


def test_mock_dashboard_steps_and_stats(client):
    stats = client.get("/api/stats").json()
    assert stats["total_steps"] == 18

    steps = client.get("/api/steps", params={"prompt_uid": "prompt-a"}).json()
    assert steps["total"] == 9
    assert steps["steps"][0]["curation"]["trainable"] is True


def test_mock_dashboard_curation_and_batch_preview(client):
    response = client.post(
        "/api/curation",
        json={
            "updates": [
                {
                    "step_key": "traj-a-0:0",
                    "quality": "good",
                    "trainable": False,
                    "tags": ["reviewed"],
                }
            ]
        },
    )
    assert response.json()["updated_count"] == 1

    curated = client.get("/api/steps", params={"quality": "good", "trainable": False}).json()
    assert curated["total"] == 1

    batch = client.post(
        "/api/batch-preview",
        json={"algorithm": "GRPO", "batch_size": 2, "n_rollouts": 2},
    ).json()
    assert batch["algorithm"] == "GRPO"
    assert batch["manifest"]["channel"] == "train"
    assert batch["prompt_groups"]


def test_mock_dashboard_prefix_tree_preview(client):
    preview = client.get("/api/prefix-tree-preview", params={"prompt_uid": "prompt-a"}).json()
    assert preview["sequence_count"] == 9
    assert preview["packed_tokens"] < preview["original_tokens"]
    assert preview["token_ratio"] < 0.25
    assert preview["nodes"][0]["start_pos"] == 0
