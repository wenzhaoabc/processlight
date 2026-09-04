from __future__ import annotations


import re

from dataclasses import dataclass

from enum import Enum

from typing import Any


from rltsc.verify.resolver import StateResolver

from rltsc.verify.utils import UNKNOWN, equal_value, norm_key, norm_text, number, order_value, parse_value, same_token


CELL_REF = r"(\[[^\]]+\]\[[^\]]+\](?:\[[^\]]+\])?)"


class ClaimStatus(str, Enum):
    CORRECT = "correct"

    INCORRECT = "incorrect"

    UNKNOWN = "unknown"


@dataclass
class ClaimVerdict:
    claim: str

    status: ClaimStatus

    reason: str

    parsed: dict[str, Any]


def verify_claim(claim: str, prompt_state_dict: dict[str, Any], prompt: str | None = None) -> ClaimVerdict:
    verifier = _ClaimVerifier(prompt_state_dict, prompt)

    return verifier.verify(claim)


def verify_claims(claims: list[str], prompt_state_dict: dict[str, Any], prompt: str | None = None) -> list[ClaimVerdict]:
    verifier = _ClaimVerifier(prompt_state_dict, prompt)

    return [verifier.verify(claim) for claim in claims]


class _ClaimVerifier(StateResolver):
    def __init__(self, prompt_state_dict: dict[str, Any], prompt: str | None):
        self.state = prompt_state_dict

        self.prompt = prompt

        self.sections = {
            "movement": self.state["movement"],
            "phase": self.state["phase"],
            "neighbor": self.state["neighbor"],
            "history": self.state["history"],
        }

        if "simulation" in self.state:
            self.sections["simulation"] = self.state["simulation"]

    def verify(self, claim: str) -> ClaimVerdict:
        text = claim.strip()

        if not text:
            return self._unknown(claim, "empty claim", {})

        tag_match = re.fullmatch(r"<claim>\s*(.*?)\s*</claim>", text, flags=re.IGNORECASE | re.DOTALL)

        if tag_match:
            text = tag_match.group(1).strip()

        verdicts = []

        for checker in [
            self._check_adj_aggregate,
            self._check_trend_value,
            self._check_trend,
            self._check_quantifier,
            self._check_rank,
            self._check_max_min,
            self._check_cell_relation,
            self._check_cell_comparison,
            self._check_cell_assignment,
        ]:
            verdict = checker(claim, text)

            if verdict is not None:
                if verdict.status is ClaimStatus.CORRECT:
                    return verdict

                verdicts.append(verdict)

        incorrect = [verdict for verdict in verdicts if verdict.status is ClaimStatus.INCORRECT]

        if incorrect:
            return incorrect[0]

        if verdicts:
            return verdicts[0]

        return self._unknown(claim, "unsupported claim format", {})

    def _check_cell_assignment(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(rf"\s*{CELL_REF}\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        left = self._cell_ref(match.group(1))

        actual = self._resolve_cell(left["row"], left["col"], left["index"])

        right_text = match.group(2)

        if re.search(r"(>=|<=|>|<)", right_text):
            return self._unknown(claim, "unsupported assignment comparison format", {"left": left, "right": right_text})

        expected = parse_value(right_text)

        if actual is UNKNOWN:
            return self._unknown(claim, "cell reference not found", {"left": left})

        return self._equal_verdict(claim, actual, expected, {"op": "=", "left": left})

    def _check_cell_relation(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(rf"\s*{CELL_REF}\s*(>=|<=|>|<)\s*{CELL_REF}\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        left = self._cell_ref(match.group(1))

        right = self._cell_ref(match.group(3))

        left_value = self._resolve_cell(left["row"], left["col"], left["index"])

        right_value = self._resolve_cell(right["row"], right["col"], right["index"])

        if left_value is UNKNOWN or right_value is UNKNOWN:
            return self._unknown(claim, "cell reference not found", {"left": left, "right": right})

        return self._order_verdict(claim, left_value, right_value, match.group(2), {"left": left, "right": right})

    def _check_cell_comparison(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(rf"\s*{CELL_REF}\s*(>=|<=|>|<)\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        left = self._cell_ref(match.group(1))

        actual = self._resolve_cell(left["row"], left["col"], left["index"])

        if actual is UNKNOWN:
            return self._unknown(claim, "cell reference not found", {"left": left})

        right_text = match.group(3)

        if "threshold" in norm_text(right_text):
            right = self._threshold_for(left["col"], right_text)

            if right is UNKNOWN:
                return self._unknown(claim, "threshold is ambiguous", {"left": left, "right": right_text})

        else:
            right = parse_value(right_text)

        return self._order_verdict(claim, actual, right, match.group(2), {"left": left, "right": right})

    def _check_max_min(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(r"\s*(MAX|MIN)\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        scope, indicator = self._extreme_scope_indicator(match.group(2), match.group(3))

        extreme = self._extreme(scope, indicator, match.group(1))

        if extreme is UNKNOWN:
            return self._unknown(claim, "extreme reference not found", {"scope": match.group(3), "indicator": match.group(2)})

        expected = match.group(4).strip()

        if self._find_key(self.sections[extreme["section"]], expected) is None:
            return self._unknown(claim, "extreme target row not found", {"extreme": extreme, "expected": expected})

        status = ClaimStatus.CORRECT if any(same_token(row, expected) for row in extreme["rows"]) else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"extreme rows are {extreme['rows']}", {"extreme": extreme, "expected": expected})

    def _extreme_scope_indicator(self, first: str, second: str) -> tuple[str, str]:
        if self._section_name(first) is not None:
            return first, second

        return second, first

    def _check_rank(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(r"\s*RANK\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        first = match.group(1)

        second = match.group(2)

        section_name = self._section_name(second)

        indicator = first

        if section_name is None:
            section_name = self._section_name(first)

            indicator = second

        if section_name is None:
            return self._unknown(claim, "rank reference not found", {"indicator": first, "scope": second})

        expected_rows = [
            re.sub(r"\s+(lowest|highest|minimum|maximum|best|first)\s*$", "", part.strip(), flags=re.IGNORECASE) for part in re.split(r">|,", match.group(3)) if part.strip()
        ]

        if not expected_rows or any(self._find_key(self.sections[section_name], row) is None for row in expected_rows):
            return self._unknown(claim, "rank target row not found", {"expected": expected_rows, "scope": section_name})

        ranking = self._rank_rows(section_name, indicator)

        if ranking is UNKNOWN:
            return self._unknown(claim, "rank reference not found", {"indicator": indicator, "scope": section_name})

        rank_groups = self._rank_groups(section_name, indicator)

        if rank_groups is UNKNOWN:
            return self._unknown(claim, "rank reference not found", {"indicator": indicator, "scope": section_name})

        status = ClaimStatus.CORRECT if self._rank_prefix_matches(rank_groups, expected_rows) else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"rank is {' > '.join(ranking)}", {"expected": expected_rows, "actual": ranking, "rank_groups": rank_groups})

    def _rank_prefix_matches(self, rank_groups: list[list[str]], expected_rows: list[str]) -> bool:
        expected_index = 0

        for group in rank_groups:
            group_expected = expected_rows[expected_index : expected_index + len(group)]

            if not group_expected:
                return True

            if len(group_expected) < len(group):
                return all(any(same_token(actual, expected) for actual in group) for expected in group_expected)

            if {norm_key(row) for row in group_expected} != {norm_key(row) for row in group}:
                return False

            expected_index += len(group)

            if expected_index == len(expected_rows):
                return True

        return expected_index == len(expected_rows)

    def _check_quantifier(self, claim: str, text: str) -> ClaimVerdict | None:
        all_match = re.fullmatch(r"\s*All\s+\[([^\]]+)\]\[([^\]]+)\]\s*<=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        no_match = re.fullmatch(r"\s*No\s+\[([^\]]+)\]\s+has\s+\[([^\]]+)\]\s*>=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not all_match and not no_match:
            return None

        match = all_match if all_match else no_match

        section_name = self._section_name(match.group(1))

        if section_name is None:
            return self._unknown(claim, "quantifier section not found", {"group": match.group(1)})

        values = self._row_metric_values(section_name, match.group(2))

        if not values:
            return self._unknown(claim, "quantifier values not found", {"group": match.group(1), "column": match.group(2)})

        right_text = match.group(3)

        right = self._threshold_for(match.group(2), right_text) if "threshold" in norm_text(right_text) else parse_value(right_text)

        if right is UNKNOWN:
            return self._unknown(claim, "threshold is ambiguous", {"column": match.group(2), "right": right_text})

        if all_match:
            ok = all(order_value(value, right, "<=") for _, value in values)

        else:
            ok = not any(order_value(value, right, ">=") for _, value in values)

        status = ClaimStatus.CORRECT if ok else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"quantifier values are {values}", {"values": values, "right": right})

    def _check_trend(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(r"\s*TREND\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(stable|increasing|decreasing)(?:\s*\([^)]*\))?\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        actual = self._trend(match.group(1), match.group(2))

        if actual is UNKNOWN:
            return self._unknown(claim, "trend reference not found", {"phase": match.group(1), "indicator": match.group(2)})

        expected = match.group(3).lower()

        status = ClaimStatus.CORRECT if actual == expected else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"trend is {actual}", {"expected": expected, "actual": actual})

    def _check_trend_value(self, claim: str, text: str) -> ClaimVerdict | None:
        match = re.fullmatch(r"\s*TREND\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        actual = self._history_value(match.group(1), match.group(2))

        if actual is UNKNOWN:
            actual = self._resolve_cell(match.group(1), match.group(2))

        if actual is UNKNOWN:
            return self._unknown(claim, "trend reference not found", {"phase": match.group(1), "indicator": match.group(2)})

        expected = parse_value(match.group(3))

        return self._equal_verdict(claim, actual, expected, {"phase": match.group(1), "indicator": match.group(2)})

    def _check_adj_aggregate(self, claim: str, text: str) -> ClaimVerdict | None:
        row_col_match = re.fullmatch(r"\s*ADJ\[([^\]]+)\]\[([A-Z]{2})\]\[([^\]]+)\]\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if row_col_match:
            values = self._adj_values(row_col_match.group(1), row_col_match.group(3), [row_col_match.group(2)])

            if values is UNKNOWN:
                return self._unknown(claim, "adjacent reference not found", {"direction": row_col_match.group(1)})

            return self._equal_verdict(claim, values[0], parse_value(row_col_match.group(4)), {"values": values})

        col_row_match = re.fullmatch(r"\s*ADJ\[([^\]]+)\]\[([^\]]+)\]\[([A-Z]{2})\]\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if col_row_match:
            values = self._adj_values(col_row_match.group(1), col_row_match.group(2), [col_row_match.group(3)])

            if values is UNKNOWN:
                return self._unknown(claim, "adjacent reference not found", {"direction": col_row_match.group(1)})

            return self._equal_verdict(claim, values[0], parse_value(col_row_match.group(4)), {"values": values})

        compare_match = re.fullmatch(r"\s*ADJ\[([^\]]+)\]\[([^\]]+)\]\s*(>=|<=|>|<)\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if compare_match:
            values = self._adj_values(compare_match.group(1), compare_match.group(2), None)

            if values is UNKNOWN:
                return self._unknown(claim, "adjacent reference not found", {"direction": compare_match.group(1), "indicator": compare_match.group(2)})

            right_text = compare_match.group(4)

            right = self._threshold_for(compare_match.group(2), right_text) if "threshold" in norm_text(right_text) else parse_value(right_text)

            if right is UNKNOWN:
                return self._unknown(claim, "threshold is ambiguous", {"indicator": compare_match.group(2), "right": right_text})

            status = ClaimStatus.CORRECT if all(order_value(value, right, compare_match.group(3)) for value in values) else ClaimStatus.INCORRECT

            return ClaimVerdict(claim, status, f"adjacent values are {values}", {"values": values, "right": right})

        match = re.fullmatch(r"\s*ADJ\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(.+?)\s*", text, flags=re.IGNORECASE)

        if not match:
            return None

        values = self._adj_values(match.group(1), match.group(2), None)

        if values is UNKNOWN:
            return self._unknown(claim, "adjacent reference not found", {"direction": match.group(1), "indicator": match.group(2)})

        if re.search(r"\s+for\s+", match.group(3), flags=re.IGNORECASE):
            return self._unknown(claim, "unsupported adjacent row filter", {"right": match.group(3)})

        expected = parse_value(match.group(3))

        status = ClaimStatus.CORRECT if all(equal_value(value, expected) for value in values) else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"adjacent values are {values}", {"values": values, "expected": expected})

    def _equal_verdict(self, claim: str, actual: Any, expected: Any, parsed: dict[str, Any]) -> ClaimVerdict:
        status = ClaimStatus.CORRECT if equal_value(actual, expected) else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"actual={actual}, expected={expected}", parsed)

    def _order_verdict(self, claim: str, left: Any, right: Any, op: str, parsed: dict[str, Any]) -> ClaimVerdict:
        left_number = number(left)

        right_number = number(right)

        if left_number is UNKNOWN or right_number is UNKNOWN:
            return self._unknown(claim, "non-numeric comparison", {**parsed, "left_value": left, "right_value": right})

        status = ClaimStatus.CORRECT if order_value(left_number, right_number, op) else ClaimStatus.INCORRECT

        return ClaimVerdict(claim, status, f"{left_number} {op} {right_number}", {**parsed, "left_value": left_number, "right_value": right_number, "op": op})

    def _unknown(self, claim: str, reason: str, parsed: dict[str, Any]) -> ClaimVerdict:
        return ClaimVerdict(claim, ClaimStatus.UNKNOWN, reason, parsed)
