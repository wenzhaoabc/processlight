from __future__ import annotations

from typing import Any

import numpy as np

STEP_METRIC_NAMES = ("phase_mapping", "local_demand", "downstream", "neighbor_coordination", "flow_trend", "fairness")


def compute_processlight_reward_metrics(non_tensor_batch: dict[str, Any]) -> dict[str, float]:
    if "score" not in non_tensor_batch:
        return {}

    score_values = [float(value) for value in non_tensor_batch["score"]]

    metrics = {
        "processlight/reward/score_mean": float(np.mean(score_values)),
        "processlight/rollout/num_rollouts": float(len(score_values)),
    }

    scalar_keys = {
        "full_reward": "full",
        "outcome_reward": "outcome",
        "phase_return_reward": "phase_return",
        "format_reward": "format",
        "valid_format": "valid_format",
        "claim_error_count": "claim_error_count",
    }

    for key, metric_name in scalar_keys.items():
        if key in non_tensor_batch:
            metrics[f"processlight/reward/{metric_name}_mean"] = float(np.mean([float(value) for value in non_tensor_batch[key]]))

    if "step_quality" in non_tensor_batch:
        step_quality = np.array(non_tensor_batch["step_quality"], dtype=float)

        evidence_quality = step_quality[:, : len(STEP_METRIC_NAMES)]

        metrics["processlight/reward/step_quality_mean"] = float(np.mean(evidence_quality))

        for index, name in enumerate(STEP_METRIC_NAMES):
            metrics[f"processlight/reward/step_quality/{name}_mean"] = float(np.mean(evidence_quality[:, index]))

    return metrics


def compute_processlight_seen_rollouts(global_steps: int, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
    if "score" not in non_tensor_batch:
        return {}

    return {"processlight/rollout/seen_rollouts": float((global_steps + 1) * len(non_tensor_batch["score"]))}
