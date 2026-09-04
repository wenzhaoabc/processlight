from __future__ import annotations


from collections import defaultdict


from dataclasses import dataclass


from typing import Any


import numpy as np


import torch


try:
    from verl.trainer.ppo.core_algos import register_adv_est


except ModuleNotFoundError:

    def register_adv_est(_name):
        """测试环境未安装verl依赖时使用的空注册器。"""

        def decorator(fn):
            return fn

        return decorator


from rltsc.trainer.parser import EVIDENCE_STEP_NAMES, STEP_DISPLAY_NAMES


@dataclass
class TreeNode:
    step_idx: int

    path: tuple[str, ...]

    rows: list[int]

    global_value: float = 0.0

    step_value: float = 0.0

    global_adv: float = 0.0

    step_adv: float = 0.0


def _cfg_value(config: Any, name: str, default: float | bool) -> float | bool:
    if isinstance(config, dict) and "processlight" in config and name in config["processlight"]:
        return config["processlight"][name]

    if config is not None and "processlight" in config and name in config.processlight:
        return config.processlight[name]

    return default


def _zscore(values: list[float], epsilon: float) -> list[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]

    tensor = torch.tensor(values, dtype=torch.float32)

    std = torch.std(tensor, unbiased=False).item()

    if std <= epsilon:
        return [0.0 for _ in values]

    mean = torch.mean(tensor).item()

    return [(value - mean) / (std + epsilon) for value in values]


def _group_rows(index: np.ndarray) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)

    for row, uid in enumerate(index):
        groups[uid].append(row)

    return groups


def _validate_step_importance(step_importance: list[Any], evidence_count: int) -> None:
    for row, values in enumerate(step_importance):
        if len(values) != evidence_count + 1:
            raise ValueError(f"Row {row} step_importance length must be {evidence_count + 1}, got {len(values)}")

        total = sum(float(values[index]) for index in range(evidence_count))

        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Row {row} evidence step_importance must sum to 1.0, got {total:.6f}")

        if float(values[evidence_count]) != 0.0:
            raise ValueError(f"Row {row} Final Answer importance must be 0")


def _build_tree(rows: list[int], tree_keys: list[list[str]], step_quality: list[list[float]], leaf_values: list[float]) -> dict[tuple[int, tuple[str, ...]], TreeNode]:
    nodes: dict[tuple[int, tuple[str, ...]], TreeNode] = {}

    for row in rows:
        path: tuple[str, ...] = ()

        for step_idx in range(len(EVIDENCE_STEP_NAMES)):
            path = (*path, tree_keys[row][step_idx])

            node_key = (step_idx, path)

            if node_key not in nodes:
                nodes[node_key] = TreeNode(step_idx=step_idx, path=path, rows=[])

            nodes[node_key].rows.append(row)

    last_step = len(EVIDENCE_STEP_NAMES) - 1

    for step_idx in range(last_step, -1, -1):
        for node in [item for item in nodes.values() if item.step_idx == step_idx]:
            node.step_value = float(np.mean([step_quality[row][step_idx] for row in node.rows]))

            if step_idx == last_step:
                node.global_value = float(np.mean([leaf_values[row] for row in node.rows]))

            else:
                children = [child for child in nodes.values() if child.step_idx == step_idx + 1 and child.path[: len(node.path)] == node.path]

                node.global_value = float(np.mean([child.global_value for child in children]))

    return nodes


def _assign_tree_advantages(nodes: dict[tuple[int, tuple[str, ...]], TreeNode], epsilon: float) -> None:
    siblings: dict[tuple[int, tuple[str, ...]], list[TreeNode]] = defaultdict(list)

    for node in nodes.values():
        siblings[(node.step_idx, node.path[:-1])].append(node)

    for group in siblings.values():
        global_advs = _zscore([node.global_value for node in group], epsilon)

        step_advs = _zscore([node.step_value for node in group], epsilon)

        for node, global_adv, step_adv in zip(group, global_advs, step_advs, strict=True):
            node.global_adv = global_adv

            node.step_adv = step_adv


def _write_span(advantages: torch.Tensor, response_mask: torch.Tensor, row: int, span: list[int], value: float) -> None:
    start, end = int(span[0]), int(span[1])

    valid_len = int(response_mask[row].sum().item())

    start = min(start, valid_len)

    end = min(end, valid_len)

    if end > start:
        advantages[row, start:end] = value


def _write_final_answer(advantages: torch.Tensor, response_mask: torch.Tensor, row: int, span: list[int], value: float) -> None:
    start, end = int(span[0]), int(span[1])

    valid_len = int(response_mask[row].sum().item())

    if end > start:
        _write_span(advantages, response_mask, row, span, value)

    elif valid_len > 0:
        advantages[row, valid_len - 1] = value


@register_adv_est("processlight_tree")
def compute_processlight_tree_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    non_tensor_batch: dict[str, Any] | None = None,
    batch: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if non_tensor_batch is None:
        raise ValueError("processlight_tree requires non_tensor_batch")

    eta = float(_cfg_value(config, "eta", 0.7))

    lambda_step = float(_cfg_value(config, "lambda_step", 0.3))

    format_coef = float(_cfg_value(config, "format_coef", 0.8))

    use_tree = bool(_cfg_value(config, "use_tree", True))

    use_importance = bool(_cfg_value(config, "use_importance", True))

    use_step_quality = bool(_cfg_value(config, "use_step_quality", True))

    step_spans = list(non_tensor_batch["step_spans"])

    step_quality = list(non_tensor_batch["step_quality"])

    tree_keys = list(non_tensor_batch["tree_node_keys"])

    step_importance = list(non_tensor_batch["step_importance"])

    outcome_reward = [float(value) for value in non_tensor_batch["outcome_reward"]]

    format_reward = [float(value) for value in non_tensor_batch["format_reward"]]

    leaf_values = [outcome + format_coef * fmt for outcome, fmt in zip(outcome_reward, format_reward, strict=True)]

    if not use_step_quality:
        step_reward = [float(np.mean([float(step_quality[row][step_idx]) for step_idx in range(len(EVIDENCE_STEP_NAMES))])) for row in range(len(step_quality))]

        leaf_values = [value + step for value, step in zip(leaf_values, step_reward, strict=True)]

    advantages = torch.zeros_like(token_level_rewards)

    grouped_rows = _group_rows(index)

    final_step_idx = len(STEP_DISPLAY_NAMES) - 1

    evidence_count = len(EVIDENCE_STEP_NAMES)

    _validate_step_importance(step_importance, evidence_count)

    for rows in grouped_rows.values():
        final_values = [leaf_values[row] for row in rows]

        final_advs = _zscore(final_values, epsilon) if norm_adv_by_std_in_grpo else [value - float(np.mean(final_values)) for value in final_values]

        if use_tree:
            nodes = _build_tree(rows, tree_keys, step_quality, leaf_values)

            _assign_tree_advantages(nodes, epsilon)

            for node in nodes.values():
                for row in node.rows:
                    imp = step_importance[row][node.step_idx]

                    global_weight = (1 - eta) + eta * evidence_count * imp if use_importance else 1.0

                    step_part = lambda_step * node.step_adv if use_step_quality else 0.0

                    _write_span(advantages, response_mask, row, step_spans[row][node.step_idx], global_weight * node.global_adv + step_part)

        else:
            for step_idx in range(evidence_count):
                global_advs = final_advs

                step_advs = _zscore([step_quality[row][step_idx] for row in rows], epsilon)

                for row, global_adv, step_adv in zip(rows, global_advs, step_advs, strict=True):
                    imp = step_importance[row][step_idx]

                    global_weight = (1 - eta) + eta * evidence_count * imp if use_importance else 1.0

                    step_part = lambda_step * step_adv if use_step_quality else 0.0

                    _write_span(advantages, response_mask, row, step_spans[row][step_idx], global_weight * global_adv + step_part)

        for row, final_adv in zip(rows, final_advs, strict=True):
            _write_final_answer(advantages, response_mask, row, step_spans[row][final_step_idx], final_adv)

    advantages = advantages * response_mask

    return advantages, advantages
