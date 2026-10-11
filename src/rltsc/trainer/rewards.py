from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rltsc.trainer.parser import EVIDENCE_STEP_NAMES, STEP_DISPLAY_NAMES, normalize_claim_text, parse_processlight_response
from rltsc.verify import ClaimStatus, ClaimVerdict, verify_claims


@dataclass(frozen=True)
class StepReward:
    quality: float

    node_key: str

    error_count: int

    unknown_count: int


@dataclass(frozen=True)
class ProcessLightReward:
    score: float

    outcome_reward: float

    phase_return_reward: float

    format_reward: float

    step_quality: list[float]

    tree_node_keys: list[str]

    parsed_phase: str

    valid_format: bool

    claim_error_count: int


def _norm_ref(ref: dict[str, Any]) -> tuple[str, str, str]:
    index = "" if ref["index"] is None else normalize_claim_text(str(ref["index"]))

    return normalize_claim_text(str(ref["row"])), normalize_claim_text(str(ref["col"])), index


def _norm_value(value: Any) -> Any:
    if isinstance(value, int | float):
        return round(float(value), 6)

    if isinstance(value, list | tuple):
        return tuple(_norm_value(item) for item in value)

    return normalize_claim_text(str(value))


def _compact_claim_text(claim: str) -> str:
    text = normalize_claim_text(claim)

    text = text.replace("minutes", "min").replace("minute", "min").replace("vehicles", "").replace("vehicle", "")

    return "".join(text.split())


def _claim_signature(verdict: ClaimVerdict) -> tuple[Any, ...]:
    parsed = verdict.parsed

    if "left" in parsed and "op" in parsed:
        if "right" in parsed and isinstance(parsed["right"], dict):
            return "cell_relation", _norm_ref(parsed["left"]), parsed["op"], _norm_ref(parsed["right"])

        if "right_value" in parsed:
            return "cell_compare", _norm_ref(parsed["left"]), parsed["op"], _norm_value(parsed["right_value"])

        return "cell_assign", _norm_ref(parsed["left"]), parsed["op"]

    if "extreme" in parsed:
        extreme = parsed["extreme"]

        return "extreme", normalize_claim_text(str(extreme["section"])), normalize_claim_text(str(extreme["row"])), _norm_value(extreme["value"])

    if "phase" in parsed and "indicator" in parsed:
        return "trend_value", normalize_claim_text(str(parsed["phase"])), normalize_claim_text(str(parsed["indicator"]))

    if "values" in parsed and "right" in parsed:
        return "aggregate_compare", _norm_value(parsed["values"]), _norm_value(parsed["right"])

    if "values" in parsed and "expected" in parsed:
        return "aggregate_assign", _norm_value(parsed["values"]), _norm_value(parsed["expected"])

    return "text", _compact_claim_text(verdict.claim)


def _claim_covers(output: ClaimVerdict, oracle: ClaimVerdict) -> bool:
    if _compact_claim_text(output.claim) == _compact_claim_text(oracle.claim):
        return True

    if "actual" in output.parsed and "expected" in output.parsed and "actual" in oracle.parsed and "expected" in oracle.parsed:
        output_expected = tuple(normalize_claim_text(str(row)) for row in output.parsed["expected"])

        oracle_expected = tuple(normalize_claim_text(str(row)) for row in oracle.parsed["expected"])

        return output_expected[: len(oracle_expected)] == oracle_expected

    return _claim_signature(output) == _claim_signature(oracle)


def semantic_claim_coverage(output_verdicts: list[ClaimVerdict], oracle_claims: list[str], prompt_state_dict: dict[str, Any]) -> float:
    if not oracle_claims:
        return 1.0

    correct_outputs = [verdict for verdict in output_verdicts if verdict.status is ClaimStatus.CORRECT]

    oracle_verdicts = verify_claims(oracle_claims, prompt_state_dict)

    for verdict in oracle_verdicts:
        if verdict.status is not ClaimStatus.CORRECT:
            raise ValueError(f"Invalid oracle claim: {verdict.claim} ({verdict.status}: {verdict.reason})")

    matched = 0

    for oracle in oracle_verdicts:
        if any(_claim_covers(output, oracle) for output in correct_outputs):
            matched += 1

    return matched / len(oracle_verdicts)


def evaluate_step_claims(step_claims: list[str], oracle_claims: list[str], prompt_state_dict: dict[str, Any]) -> StepReward:
    verdicts = verify_claims(step_claims, prompt_state_dict) if step_claims else []

    error_count = sum(1 for verdict in verdicts if verdict.status is ClaimStatus.INCORRECT)

    unknown_count = sum(1 for verdict in verdicts if verdict.status is ClaimStatus.UNKNOWN)

    if error_count or unknown_count:
        quality = -1.0

    else:
        quality = semantic_claim_coverage(verdicts, oracle_claims, prompt_state_dict)

    node_key = f"quality:{quality:.6f}"

    return StepReward(quality=quality, node_key=node_key, error_count=error_count, unknown_count=unknown_count)


def normalized_phase_return(parsed_phase: str, phase_return: dict[str, Any], invalid_phase_reward: float) -> float:
    if not parsed_phase or parsed_phase not in phase_return:
        return invalid_phase_reward

    values = [float(value) for value in phase_return.values()]

    min_return = min(values)

    max_return = max(values)

    if max_return == min_return:
        return 0.5

    return (float(phase_return[parsed_phase]) - min_return) / (max_return - min_return)


def compute_processlight_reward(
    response: str,
    solution: dict[str, Any],
    extra_info: dict[str, Any],
    invalid_phase_reward: float,
    format_coef: float,
) -> ProcessLightReward:
    parsed = parse_processlight_response(response)

    if parsed.phase:
        outcome_reward = 1.0 if parsed.phase == solution["best_phase"] else 0.0

    else:
        outcome_reward = 0.0

    phase_return_reward = normalized_phase_return(parsed.phase, solution["phase_return"], invalid_phase_reward)

    format_reward = 1.0 if parsed.valid_format else 0.0

    step_quality: list[float] = []

    tree_node_keys: list[str] = []

    claim_error_count = 0

    for name in STEP_DISPLAY_NAMES:
        if name == "Final Answer":
            step_quality.append(1.0)

            tree_node_keys.append("final_answer")

            continue

        oracle_claims = extra_info["oracle_claims"][name]

        step_reward = evaluate_step_claims(parsed.claims[name], oracle_claims, extra_info["prompt_state_dict"])

        step_quality.append(step_reward.quality)

        tree_node_keys.append(step_reward.node_key)

        claim_error_count += step_reward.error_count + step_reward.unknown_count

    score = outcome_reward + format_coef * format_reward

    score += sum(step_quality[: len(EVIDENCE_STEP_NAMES)]) / len(EVIDENCE_STEP_NAMES)

    return ProcessLightReward(
        score=score,
        outcome_reward=outcome_reward,
        phase_return_reward=phase_return_reward,
        format_reward=format_reward,
        step_quality=step_quality,
        tree_node_keys=tree_node_keys,
        parsed_phase=parsed.phase,
        valid_format=parsed.valid_format,
        claim_error_count=claim_error_count,
    )
