from __future__ import annotations


import re


from dataclasses import dataclass


STEP_DISPLAY_NAMES = (
    "Phase Mapping",
    "Local Demand",
    "Downstream",
    "Neighbor Coordination",
    "Flow Trend",
    "Fairness",
    "Final Answer",
)


EVIDENCE_STEP_NAMES = STEP_DISPLAY_NAMES[:-1]


PHASE_ORDER = ("ETWT", "NTST", "ELWL", "NLSL")


def parse_phase(text: str) -> str:
    matches = re.findall(r"<phase>(.*?)</phase>", text, flags=re.IGNORECASE | re.DOTALL)

    if not matches:
        raise ValueError("No <phase>...</phase> tag found")

    phase = matches[-1].strip()

    if phase not in PHASE_ORDER:
        raise ValueError(f"Invalid phase: {phase}")

    return phase


def extract_claim_tags(text: str) -> list[str]:
    claims = re.findall(r"<claim>(.*?)</claim>", text, flags=re.IGNORECASE | re.DOTALL)

    return [claim.strip() for claim in claims if claim.strip()]


@dataclass(frozen=True)
class StepSection:
    name: str

    text: str

    start: int

    end: int


@dataclass(frozen=True)
class ParsedResponse:
    sections: dict[str, StepSection]

    claims: dict[str, list[str]]

    phase: str

    valid_format: bool

    format_errors: list[str]


def normalize_claim_text(text: str) -> str:
    lowered = text.strip().lower()

    return re.sub(r"\s+", " ", lowered)


def parse_sections(response: str) -> tuple[dict[str, StepSection], list[str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", response))

    sections: dict[str, StepSection] = {}

    errors: list[str] = []

    expected = list(STEP_DISPLAY_NAMES)

    observed = [match.group(1).strip() for match in matches]

    if observed[: len(expected)] != expected:
        errors.append(f"Unexpected section order: {observed}")

    for index, match in enumerate(matches):
        name = match.group(1).strip()

        if name not in STEP_DISPLAY_NAMES:
            continue

        body_start = match.end()

        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(response)

        sections[name] = StepSection(name=name, text=response[body_start:body_end].strip(), start=match.start(), end=body_end)

    for name in STEP_DISPLAY_NAMES:
        if name not in sections:
            errors.append(f"Missing section: {name}")

        elif name != "Final Answer" and ("<claim>" not in sections[name].text or "</claim>" not in sections[name].text):
            errors.append(f"Missing claim tag: {name}")

    return sections, errors


def parse_processlight_response(response: str) -> ParsedResponse:
    sections, errors = parse_sections(response)

    claims: dict[str, list[str]] = {}

    for name in STEP_DISPLAY_NAMES:
        claims[name] = extract_claim_tags(sections[name].text) if name in sections else []

    if "Final Answer" not in sections:
        phase = ""

    else:
        try:
            phase = parse_phase(sections["Final Answer"].text)

        except Exception as exc:
            phase = ""

            errors.append(str(exc))

    if phase and phase not in PHASE_ORDER:
        errors.append(f"Invalid phase: {phase}")

    return ParsedResponse(sections=sections, claims=claims, phase=phase, valid_format=not errors, format_errors=errors)


def char_span_to_token_span(text: str, char_start: int, char_end: int, tokenizer) -> tuple[int, int]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    offsets = encoded["offset_mapping"]

    token_start = None

    token_end = None

    for index, (start, end) in enumerate(offsets):
        if token_start is None and end > char_start:
            token_start = index

        if start < char_end:
            token_end = index + 1

    if token_start is None or token_end is None or token_end <= token_start:
        raise ValueError(f"Cannot map char span to token span: {char_start}-{char_end}")

    return token_start, token_end
