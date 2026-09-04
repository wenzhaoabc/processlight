from __future__ import annotations


import argparse


import json


import random


from pathlib import Path


from typing import Any


import pandas as pd


from rltsc.processlight_contract import validate_training_prompt_state


PHASE_ORDER = ("ETWT", "NTST", "ELWL", "NLSL")


SFT_TARGET_ROWS = 500


GRPO_TARGET_ROWS = 3000


BALANCE_SEED = 202605


def read_json_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if path.suffix != ".json":
        raise ValueError(f"Input must be .json: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"JSON payload must be a list: {path}")

    return payload


def write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_parquet(path, index=False)


def sft_phase(row: dict[str, Any]) -> str:
    if "extra_info" in row:
        return row["extra_info"]["best_phase"]

    return row["metadata"]["best_phase"]


def grpo_phase(row: dict[str, Any]) -> str:
    return row["solution"]["best_phase"]


def balanced_rows(rows: list[dict[str, Any]], target: int, phase_reader) -> list[dict[str, Any]]:
    rng = random.Random(BALANCE_SEED)

    grouped = {phase: [] for phase in PHASE_ORDER}

    for row in rows:
        phase = phase_reader(row)

        if phase not in grouped:
            raise ValueError(f"Invalid best_phase: {phase}")

        grouped[phase].append(row)

    for phase in PHASE_ORDER:
        rng.shuffle(grouped[phase])

    quota = target // len(PHASE_ORDER)

    selected: list[dict[str, Any]] = []

    selected_ids = set()

    for phase in PHASE_ORDER:
        for row in grouped[phase][:quota]:
            selected.append(row)

            selected_ids.add(id(row))

    phase_cursor = {phase: min(quota, len(grouped[phase])) for phase in PHASE_ORDER}

    while len(selected) < target:
        changed = False

        for phase in PHASE_ORDER:
            cursor = phase_cursor[phase]

            if cursor >= len(grouped[phase]):
                continue

            row = grouped[phase][cursor]

            if id(row) not in selected_ids:
                selected.append(row)

                selected_ids.add(id(row))

            phase_cursor[phase] = cursor + 1

            changed = True

            if len(selected) == target:
                break

        if not changed:
            break

    rng.shuffle(selected)

    return selected


def phase_distribution(rows: list[dict[str, Any]], phase_reader) -> dict[str, int]:
    distribution = {phase: 0 for phase in PHASE_ORDER}

    for row in rows:
        distribution[phase_reader(row)] += 1

    return distribution


def build_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []

    for index, row in enumerate(rows):
        if "response" in row:
            completion = [{"role": "assistant", "content": row["response"]}]

        else:
            completion = row["completion"]

        if len(completion) != 1 or completion[0]["role"] != "assistant":
            raise ValueError(f"Invalid SFT completion at row {index}")

        output.append({"messages": list(row["prompt"]) + completion})

    return output


def build_grpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []

    for index, row in enumerate(rows):
        extra_info = dict(row["extra_info"])

        validate_training_prompt_state(extra_info["prompt_state_dict"])

        extra_info["index"] = index

        output.append(
            {
                "data_source": row["data_source"],
                "prompt": row["prompt"],
                "reward_model": {"ground_truth": row["solution"]},
                "extra_info": extra_info,
            }
        )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ProcessLight SFT/GRPO parquet data for verl")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    sft = subparsers.add_parser("sft", description="Convert fixed ProcessLight SFT json to verl SFT parquet")

    sft.add_argument("--input", required=True, help="datasets/process/process_sft_1k.json。")

    sft.add_argument("--output", required=True, help="输出verl SFT parquet路径。")

    grpo = subparsers.add_parser("grpo", description="Convert fixed ProcessLight GRPO json to verl RL parquet")

    grpo.add_argument("--input", required=True, help="samples_grpo.py产出的grpo_samples文件。")

    grpo.add_argument("--output", required=True, help="输出grpo_train.parquet路径。")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_json_rows(args.input)

    if args.mode == "sft":
        rows = balanced_rows(rows, SFT_TARGET_ROWS, sft_phase)

        write_parquet(args.output, build_sft_rows(rows))

    elif args.mode == "grpo":
        rows = balanced_rows(rows, GRPO_TARGET_ROWS, grpo_phase)

        write_parquet(args.output, build_grpo_rows(rows))

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    distribution = phase_distribution(rows, sft_phase if args.mode == "sft" else grpo_phase)

    print(f"phase_distribution={distribution}")

    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
