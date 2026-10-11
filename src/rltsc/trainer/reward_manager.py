from __future__ import annotations

from typing import Any

import torch

try:
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


except ModuleNotFoundError:
    DataProto = Any

    class RewardManagerBase:
        """测试环境缺少verl依赖时使用的最小基类。"""

        def __init__(self, config, tokenizer, compute_score):
            self.config = config

            self.tokenizer = tokenizer

            self.compute_score = compute_score


from rltsc.trainer.parser import EVIDENCE_STEP_NAMES, STEP_DISPLAY_NAMES, char_span_to_token_span, parse_processlight_response
from rltsc.trainer.rewards import compute_processlight_reward


class ProcessLightRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)

        self.reward_router_address = reward_router_address

        self.reward_model_tokenizer = reward_model_tokenizer

        self.invalid_phase_reward = -1.0

        self.format_coef = 0.8

        self.score_mode = "full"

        if "algorithm" in config and "processlight" in config.algorithm and "format_coef" in config.algorithm.processlight:
            self.format_coef = float(config.algorithm.processlight.format_coef)

        if "processlight" in config.reward:
            process_cfg = config.reward.processlight

            if "invalid_phase_reward" in process_cfg:
                self.invalid_phase_reward = float(process_cfg.invalid_phase_reward)

            if "score_mode" in process_cfg:
                self.score_mode = str(process_cfg.score_mode)

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        data_item = data[-1:][0]

        response_ids = data_item.batch["responses"]

        response_length = response_ids.shape[-1]

        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())

        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))

        solution = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        extra_info = data_item.non_tensor_batch["extra_info"]

        reward = compute_processlight_reward(
            response=response_str,
            solution=solution,
            extra_info=extra_info,
            invalid_phase_reward=self.invalid_phase_reward,
            format_coef=self.format_coef,
        )

        spans = self._step_token_spans(response_str, valid_response_length)

        step_importance = self._step_importance(extra_info["step_importance"])

        reward_score = self._select_reward_score(reward)

        return {
            "reward_score": reward_score,
            "reward_extra_info": {
                "score": reward_score,
                "full_reward": reward.score,
                "outcome_reward": reward.outcome_reward,
                "phase_return_reward": reward.phase_return_reward,
                "format_reward": reward.format_reward,
                "step_quality": reward.step_quality,
                "step_spans": spans,
                "tree_node_keys": reward.tree_node_keys,
                "step_importance": step_importance,
                "parsed_phase": reward.parsed_phase,
                "valid_format": reward.valid_format,
                "claim_error_count": reward.claim_error_count,
            },
        }

    def _select_reward_score(self, reward) -> float:
        if self.score_mode == "outcome":
            return reward.outcome_reward

        if self.score_mode == "phase_return":
            return reward.phase_return_reward

        if self.score_mode == "outcome_format":
            return reward.outcome_reward + self.format_coef * reward.format_reward

        if self.score_mode == "full":
            return reward.score

        raise ValueError(f"Unknown ProcessLight score_mode: {self.score_mode}")

    def _step_importance(self, importance_dict: dict[str, Any]) -> list[float]:
        if "Final Answer" in importance_dict:
            raise ValueError("Final Answer must not have step importance")

        values: list[float] = []

        for name in EVIDENCE_STEP_NAMES:
            if name not in importance_dict:
                raise ValueError(f"Missing step importance: {name}")

            values.append(float(importance_dict[name]))

        total = sum(values)

        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Step importance must sum to 1.0, got {total:.6f}")

        return values + [0.0]

    def _step_token_spans(self, response_str: str, valid_response_length: int) -> list[list[int]]:
        parsed = parse_processlight_response(response_str)

        spans: list[list[int]] = []

        for name in STEP_DISPLAY_NAMES:
            if name not in parsed.sections:
                spans.append([0, 0])

                continue

            section = parsed.sections[name]

            start, end = char_span_to_token_span(response_str, section.start, section.end, self.tokenizer)

            spans.append([min(start, valid_response_length), min(end, valid_response_length)])

        return spans

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        prompt_length = data.batch["prompts"].size(1)

        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)

        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        rows = torch.arange(rm_scores.size(0), device=rm_scores.device)

        rm_scores[rows, valid_response_length - 1] = rm_scores.new_tensor(scores)

        return rm_scores
