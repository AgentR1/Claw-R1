import os

import pytest

pytest.importorskip("verl.base_config")
ray = pytest.importorskip("ray")

from claw_r1.data_pool.data_model import DataPoolConfig, Step  # noqa: E402
from claw_r1.data_pool.data_pool import DataPool  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAY_TESTS") != "1",
    reason="Ray actor tests are opt-in because local Ray startup is environment-sensitive",
)


class FakeBackend:
    def convert(self, steps):
        return [
            {
                "trajectory_uid": step.trajectory_uid,
                "step_index": step.step_index,
                "reward": step.reward,
            }
            for step in steps
        ]


@pytest.fixture
def data_pool():
    if not ray.is_initialized():
        ray.init(address="local", include_dashboard=False, num_cpus=1, ignore_reinit_error=True)
    actor = DataPool.remote(DataPoolConfig(n_rollouts=1), FakeBackend())
    yield actor
    ray.kill(actor)


def test_list_steps_trajectories_events_and_curation(data_pool):
    steps = [
        Step(
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5],
            reward=0.5,
            trajectory_uid="traj-1",
            prompt_uid="prompt-1",
            step_index=0,
            policy_version=7,
            is_last=True,
            metadata={"agent": "agent-a", "task": "math"},
        )
    ]
    ray.get(data_pool.submit_steps.remote(steps))

    listed = ray.get(data_pool.list_steps.remote("train", prompt_uid="prompt-1"))
    assert listed["total"] == 1
    row = listed["steps"][0]
    assert row["step_key"] == "traj-1:0"
    assert row["curation"]["quality"] == "unreviewed"
    assert row["curation"]["trainable"] is True

    trajectories = ray.get(data_pool.list_trajectories.remote("train"))
    assert trajectories["trajectories"][0]["trajectory_uid"] == "traj-1"
    assert trajectories["trajectories"][0]["reward_sum"] == 0.5

    events = ray.get(data_pool.get_step_events.remote("train", cursor=0))
    assert events["events"][0]["type"] == "step_submitted"

    update = ray.get(
        data_pool.update_step_curation.remote(
            [{"step_key": "traj-1:0", "quality": "good", "trainable": False, "tags": ["keep"]}],
            "train",
        )
    )
    assert update["updated_count"] == 1

    curated = ray.get(data_pool.list_steps.remote("train", quality="good", trainable=False))
    assert curated["total"] == 1
    assert curated["steps"][0]["curation"]["tags"] == ["keep"]


def test_curation_side_table_does_not_change_fetch_batch(data_pool):
    step = Step(
        prompt_ids=[10],
        response_ids=[11],
        reward=1.0,
        trajectory_uid="traj-2",
        prompt_uid="prompt-2",
        step_index=0,
        is_last=True,
    )
    ray.get(data_pool.submit_steps.remote([step]))
    ray.get(
        data_pool.update_step_curation.remote(
            [{"step_key": "traj-2:0", "quality": "bad", "trainable": False}],
            "train",
        )
    )

    batch = ray.get(data_pool.fetch_batch.remote(batch_size=1, n_rollouts=1, channel="train"))

    assert batch == [{"trajectory_uid": "traj-2", "step_index": 0, "reward": 1.0}]
