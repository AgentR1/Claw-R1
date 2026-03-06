"""Pydantic models for Gateway Server request/response schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── Step submission ──────────────────────────────────────────────────────


class StepPayload(BaseModel):
    """Wire format for a single Step (JSON-serialisable subset of data_model.Step)."""

    prompt_ids: list[int]
    response_ids: list[int]
    multi_modal_data: Optional[dict[str, Any]] = None
    reward: Optional[float] = None
    rollout_log_probs: Optional[list[float]] = None
    routed_experts: Any = None
    trajectory_uid: str = ""
    prompt_uid: str = ""
    step_index: int = 0
    policy_version: int = 0
    is_last: bool = False
    metadata: Optional[dict[str, Any]] = None


class SubmitStepsRequest(BaseModel):
    """POST /submit_steps request body."""

    steps: list[StepPayload]


class SubmitStepsResponse(BaseModel):
    """POST /submit_steps response body."""

    accepted: int


# ── LLM generation ──────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """POST /generate request body — mirrors the old server_manager.generate() interface."""

    prompt_ids: list[int]
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    image_data: Optional[list[Any]] = None
    video_data: Optional[list[Any]] = None


class GenerateResponse(BaseModel):
    """POST /generate response body."""

    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    stop_reason: Optional[str] = None


# ── Reward computation ───────────────────────────────────────────────────


class ComputeRewardRequest(BaseModel):
    """POST /compute_reward request body.

    Carries the raw data needed to compute a reward for one sample.
    The Gateway handles padding and DataProto construction internally.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    multi_modal_data: Optional[dict[str, Any]] = None
    num_turns: int = 2
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    dataset_fields: dict[str, Any] = Field(default_factory=dict)


class ComputeRewardResponse(BaseModel):
    """POST /compute_reward response body."""

    reward_score: float
    reward_extra_info: dict[str, Any] = Field(default_factory=dict)


# ── Trajectory completion (black-box) ─────────────────────────────────────


class CompleteTrajectoryRequest(BaseModel):
    """POST /complete_trajectory/{trajectory_uid} request body."""

    reward: Optional[float] = None
    channel: str = "train"


# ── Init trajectory (black-box, online mode) ─────────────────────────────


class InitTrajectoryResponse(BaseModel):
    """POST /init_trajectory response body."""

    trajectory_uid: str
    base_url: str
