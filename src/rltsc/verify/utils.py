from __future__ import annotations

import ast
import re
from typing import Any


class UnknownValue:
    pass


UNKNOWN = UnknownValue()


ALIASES = {
    "queue": {"earlyqueuelength", "queuelength"},
    "queuelength": {"earlyqueuelength", "queue"},
    "localqueue": {"localqueuelength"},
    "localqueuelength": {"localqueue"},
    "globalqueue": {"globalqueuelength"},
    "globalqueuelength": {"globalqueue"},
    "phasepressure": {"pressure"},
    "totalvehiclecount": {"vehiclecount"},
    "phasevehiclecount": {"vehiclecount"},
    "avgwaitingtime": {"avgwaitingtimeminutes"},
    "avgwaitingtimeminutes": {"avgwaitingtime"},
    "waitingtime": {"avgwaitingtime", "avgwaitingtimeminutes"},
    "segmentvehoccupancy": {"segmentvehcount"},
    "segmentvehcount": {"segmentvehoccupancy"},
    "movement": {"movements"},
    "movements": {"movement"},
}


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def norm_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def same_token(left: str, right: str) -> bool:
    return norm_key(left) == norm_key(right)


def parse_value(value: Any) -> Any:
    if isinstance(value, int | float | list):
        return value

    text = str(value).strip()

    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()

    cleaned = re.sub(r"\s*(?:min|mins|minute|minutes|step|steps|vehicle|vehicles|veh|vehicless?)\s*$", "", text, flags=re.IGNORECASE).strip()

    threshold_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s+threshold", cleaned, flags=re.IGNORECASE)

    if threshold_match:
        cleaned = threshold_match.group(1)

    threshold_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s+(?:min|mins|minute|minutes|step|steps|vehicle|vehicles|veh)\s+threshold", cleaned, flags=re.IGNORECASE)

    if threshold_match:
        cleaned = threshold_match.group(1)

    threshold_match = re.fullmatch(r"threshold\s+([-+]?\d+(?:\.\d+)?)", cleaned, flags=re.IGNORECASE)

    if threshold_match:
        cleaned = threshold_match.group(1)

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return float(cleaned) if "." in cleaned else int(cleaned)

    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = ast.literal_eval(cleaned)

        except (ValueError, SyntaxError):
            return text

        if isinstance(parsed, list):
            return [parse_value(item) for item in parsed]

    return text


def number(value: Any) -> float | UnknownValue:
    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, str):
        parsed = parse_value(value)

        if isinstance(parsed, int | float):
            return float(parsed)

    return UNKNOWN


def equal_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(equal_value(left, right) for left, right in zip(actual, expected, strict=False))

    actual_number = number(actual)

    expected_number = number(expected)

    if actual_number is not UNKNOWN and expected_number is not UNKNOWN:
        return abs(float(actual_number) - float(expected_number)) <= 1e-6

    return norm_text(str(actual)) == norm_text(str(expected))


def order_value(left: Any, right: Any, op: str) -> bool:
    left_number = number(left)

    right_number = number(right)

    if left_number is UNKNOWN or right_number is UNKNOWN:
        return False

    if op == ">":
        return left_number > right_number

    if op == "<":
        return left_number < right_number

    if op == ">=":
        return left_number >= right_number

    if op == "<=":
        return left_number <= right_number

    return False


def sequence_values(value: Any) -> list[Any] | UnknownValue:
    if isinstance(value, list):
        return value

    if isinstance(value, str) and "->" in value:
        cleaned = value.replace("min", "")

        return [parse_value(part.strip()) for part in cleaned.split("->")]

    return UNKNOWN


def sequence_index(index_text: str, seq_len: int) -> int | UnknownValue:
    text = norm_text(index_text)

    if text in {"first", "early", "near"}:
        return 0

    if text in {"middle", "mid"}:
        return 1 if seq_len > 1 else UNKNOWN

    if text in {"last", "final", "far"}:
        return seq_len - 1

    segment_match = re.fullmatch(r"segment(\d+)", text)

    if segment_match:
        index = int(segment_match.group(1)) - 1

        return index if 0 <= index < seq_len else UNKNOWN

    if re.fullmatch(r"-?\d+", text):
        index = int(text)

        if index < 0:
            index = seq_len + index

        return index if 0 <= index < seq_len else UNKNOWN

    return UNKNOWN
