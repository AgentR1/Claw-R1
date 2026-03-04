"""Training backend interface and implementations.

The TrainingBackend is the bridge between DataPool's internal representation
(``list[Step]``) and whatever format the training engine expects.  It is
pluggable: currently ``VerlBackend`` converts to verl's ``DataProto``, but
other backends can be added without touching DataPool itself.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from tensordict import TensorDict

from claw_r1.data_pool.data_model import Step
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask


class TrainingBackend(ABC):
    """Abstract interface for converting DataPool steps into training batches."""

    @abstractmethod
    def convert(self, steps: list[Step]) -> Any:
        """Convert a list of Steps into the training engine's native batch format.

        Args:
            steps (list[Step]): Raw steps collected from the DataPool.

        Returns:
            Any: A training-ready batch in the engine's native format.
        """
        ...


class VerlBackend(TrainingBackend):
    """Convert ``list[Step]`` into verl ``DataProto``.

    Produces the same tensor layout that the existing verl training pipeline
    expects (``input_ids``, ``attention_mask``, ``position_ids``, ``prompts``,
    ``responses``, etc.), so the downstream reward / advantage / actor-update
    code can consume it without modification.

    Tensor layout (consistent with the legacy verl rollout convention):

    - prompt_ids: left-padded with pad_token_id to ``prompt_length``
    - response_ids: right-padded with pad_token_id to ``response_length``
    - input_ids: ``[prompt_ids | response_ids]``
    - attention_mask: 0 for padding, 1 for real tokens across the full sequence
    - response_mask: 1 for LLM-generated response tokens, 0 for padding
    - position_ids: sequential positions for non-padding tokens (0-based)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_length: int,
        response_length: int,
        processor: Optional[AutoProcessor] = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self._prompt_length = prompt_length
        self._response_length = response_length

    # ── Public API ─────────────────────────────────────────────────────────

    def convert(self, steps: list[Step]) -> DataProto:
        """Convert steps into a ``DataProto``.

        Each step is independently padded (prompt left-padded, response
        right-padded), then all steps are stacked into a single batch.

        Args:
            steps (list[Step]): Raw steps from the DataPool.

        Returns:
            DataProto: Training-ready batch.
        """
        if not steps:
            raise ValueError("Cannot convert an empty step list.")

        prompt_ids_list: list[torch.Tensor] = []
        response_ids_list: list[torch.Tensor] = []
        response_mask_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []
        input_ids_list: list[torch.Tensor] = []
        position_ids_list: list[torch.Tensor] = []
        logprobs_list: list[Optional[torch.Tensor]] = []
        experts_list: list[Optional[torch.Tensor]] = []
        multi_modal_inputs_list: list[Optional[dict]] = []

        has_reward = any(s.reward is not None for s in steps)
        reward_tensors: list[torch.Tensor] = []

        for step in steps:
            padded = self._pad_single_step(step)
            prompt_ids_list.append(padded["prompt_ids"])
            response_ids_list.append(padded["response_ids"])
            response_mask_list.append(padded["response_mask"])
            attention_mask_list.append(padded["attention_mask"])
            input_ids_list.append(padded["input_ids"])
            position_ids_list.append(padded["position_ids"])
            logprobs_list.append(padded.get("response_logprobs"))
            experts_list.append(padded.get("routed_experts"))
            multi_modal_inputs_list.append(padded.get("multi_modal_inputs"))

            if has_reward:
                reward_tensor = torch.zeros_like(padded["response_mask"], dtype=torch.float32)
                valid_length = int(padded["response_mask"].sum().item())
                if valid_length > 0 and step.reward is not None:
                    reward_tensor[0, valid_length - 1] = float(step.reward)
                reward_tensors.append(reward_tensor)

        return self._assemble_batch(
            steps,
            prompt_ids_list=prompt_ids_list,
            response_ids_list=response_ids_list,
            response_mask_list=response_mask_list,
            attention_mask_list=attention_mask_list,
            input_ids_list=input_ids_list,
            position_ids_list=position_ids_list,
            reward_tensors=reward_tensors if has_reward else None,
            logprobs_list=logprobs_list,
            experts_list=experts_list,
            multi_modal_inputs_list=multi_modal_inputs_list
        )

    # ── Per-step padding ───────────────────────────────────────────────────

    def _pad_single_step(self, step: Step) -> dict[str, Any]:
        """Pad a single Step's sequences and compute derived tensors.

        Returns a dict of tensors, each with a leading batch dim of 1.
        """
        self._tokenizer.padding_side = "left"
        prompt_out = self._tokenizer.pad(
            {"input_ids": step.prompt_ids},
            padding="max_length",
            max_length=self._prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_out["input_ids"].dim() == 1:
            prompt_out["input_ids"] = prompt_out["input_ids"].unsqueeze(0)
            prompt_out["attention_mask"] = prompt_out["attention_mask"].unsqueeze(0)

        self._tokenizer.padding_side = "right"
        response_out = self._tokenizer.pad(
            {"input_ids": step.response_ids},
            padding="max_length",
            max_length=self._response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_out["input_ids"].dim() == 1:
            response_out["input_ids"] = response_out["input_ids"].unsqueeze(0)
            response_out["attention_mask"] = response_out["attention_mask"].unsqueeze(0)

        # Step treats all response tokens as LLM-generated, so response_mask
        # equals the response attention_mask (1 for real tokens, 0 for padding).
        response_mask = response_out["attention_mask"].clone()

        attention_mask = torch.cat(
            [prompt_out["attention_mask"], response_out["attention_mask"]], dim=1,
        )
        input_ids = torch.cat(
            [prompt_out["input_ids"], response_out["input_ids"]], dim=1,
        )

        result: dict[str, Any] = {
            "prompt_ids": prompt_out["input_ids"],
            "response_ids": response_out["input_ids"],
            "response_mask": response_mask,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }

        # Optional: rollout log-probs
        if step.rollout_log_probs is not None:
            pad_size = self._response_length - len(step.rollout_log_probs)
            result["response_logprobs"] = torch.tensor(
                step.rollout_log_probs + [0.0] * pad_size,
            ).unsqueeze(0)

        # Optional: MoE routed experts
        if step.routed_experts is not None:
            result["routed_experts"] = self._pad_routed_experts(
                step, prompt_out["input_ids"], input_ids,
            )

        # Multi-modal inputs
        multi_modal_inputs = self._compute_multi_modal_inputs(
            step.multi_modal_data, input_ids,
        )
        if multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs

        # Position IDs (must come after multi_modal_inputs)
        result["position_ids"] = self._compute_position_ids(
            input_ids, attention_mask, multi_modal_inputs,
        )

        return result

    # ── Batch assembly ─────────────────────────────────────────────────────

    def _assemble_batch(
        self,
        steps: list[Step],
        *,
        prompt_ids_list: list[torch.Tensor],
        response_ids_list: list[torch.Tensor],
        response_mask_list: list[torch.Tensor],
        attention_mask_list: list[torch.Tensor],
        input_ids_list: list[torch.Tensor],
        position_ids_list: list[torch.Tensor],
        reward_tensors: Optional[list[torch.Tensor]],
        logprobs_list: list[Optional[torch.Tensor]],
        experts_list: list[Optional[torch.Tensor]],
        multi_modal_inputs_list: list[Optional[dict]],
    ) -> DataProto:
        """Stack per-step tensors into a single ``DataProto`` batch."""
        prompt_ids = torch.cat(prompt_ids_list, dim=0)
        response_ids = torch.cat(response_ids_list, dim=0)
        response_mask = torch.cat(response_mask_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)
        position_ids = torch.cat(position_ids_list, dim=0)

        optional_outputs: dict[str, torch.Tensor] = {}
        if reward_tensors is not None:
            optional_outputs["rm_scores"] = torch.cat(reward_tensors, dim=0)
        if all(lp is not None for lp in logprobs_list):
            optional_outputs["rollout_log_probs"] = torch.cat(logprobs_list, dim=0)
        if all(ex is not None for ex in experts_list):
            optional_outputs["routed_experts"] = torch.cat(experts_list, dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=prompt_ids.size(0),
        )

        non_tensor_batch: dict[str, np.ndarray] = {
            "prompt_uids": np.array(
                [s.prompt_uid for s in steps], dtype=object,
            ),
            "trajectory_uids": np.array(
                [s.trajectory_uid for s in steps], dtype=object,
            ),
            "step_indices": np.array(
                [s.step_index for s in steps], dtype=np.int32,
            ),
        }

        # Expand Step.metadata fields into non_tensor_batch so the Trainer
        # can access dataset info (data_source, reward_model, uid, etc.)
        # without needing the original dataset batch.
        metadata_keys: set[str] = set()
        for s in steps:
            if s.metadata:
                metadata_keys.update(s.metadata.keys())
        for key in sorted(metadata_keys):
            non_tensor_batch[key] = np.array(
                [s.metadata.get(key) if s.metadata else None for s in steps],
                dtype=object,
            )

        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(
                multi_modal_inputs_list, dtype=object,
            )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _pad_routed_experts(
        self,
        step: Step,
        padded_prompt_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pad MoE routing decisions to the full sequence length."""
        total_length = input_ids.shape[1]
        length, layer_num, topk_num = step.routed_experts.shape
        experts_tensor = torch.from_numpy(step.routed_experts)
        routed_experts = torch.zeros(
            1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype,
        )

        start_pos = padded_prompt_ids.shape[1] - len(step.prompt_ids)
        end_pos = min(start_pos + length, total_length)

        if start_pos < 0 or end_pos > total_length:
            raise ValueError(
                f"Invalid position range: start_pos={start_pos}, "
                f"end_pos={end_pos}, total_length={total_length}"
            )

        routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)
        return routed_experts

    def _compute_multi_modal_inputs(
        self,
        multi_modal_data: Optional[dict[str, Any]],
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        if self._processor is None or multi_modal_data is None:
            return {}

        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)

        current_text = self._tokenizer.decode(
            input_ids.squeeze(0), skip_special_tokens=True,
        )
        multi_modal_inputs = self._processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        return dict(multi_modal_inputs.convert_to_tensors("pt"))

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute position ids, with special handling for multi-modal models."""
        if self._processor is None or not multi_modal_inputs:
            return compute_position_id_with_mask(attention_mask)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        vision_position_ids, _ = self._processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones(
            (1, input_ids.shape[1]), dtype=torch.long,
        )
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)

        return torch.cat(
            (text_position_ids, vision_position_ids), dim=1,
        )
