from __future__ import annotations


from typing import Any


PROMPT_STATE_REQUIRED_KEYS = ("movement", "phase", "neighbor", "history", "simulation")


def attach_simulation_to_prompt_state(prompt_state_dict: dict[str, Any], simulation: dict[str, Any]) -> dict[str, Any]:
    merged = dict(prompt_state_dict)

    if "simulation" in merged and merged["simulation"] != simulation:
        raise ValueError("prompt_state_dict.simulation conflicts with top-level simulation")

    merged["simulation"] = simulation

    validate_training_prompt_state(merged)

    return merged


def validate_training_prompt_state(prompt_state_dict: dict[str, Any]) -> None:
    missing_keys = [key for key in PROMPT_STATE_REQUIRED_KEYS if key not in prompt_state_dict]

    if missing_keys:
        raise ValueError(f"prompt_state_dict missing keys: {missing_keys}")
