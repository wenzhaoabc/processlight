from __future__ import annotations


import re

from typing import Any


from rltsc.verify.utils import ALIASES, UNKNOWN, norm_key, norm_text, number, parse_value, sequence_index, sequence_values


class StateResolver:
    def _resolve_cell(self, row: str, col: str, index: str | None = None) -> Any:
        for section_name in ["movement", "phase", "history", "simulation"]:
            if section_name not in self.sections:
                continue

            section = self.sections[section_name]

            real_row = self._find_key(section, row)

            if real_row is None or not isinstance(section[real_row], dict):
                continue

            if section_name == "phase" and norm_key(col) == "phase":
                return real_row

            real_col = self._find_key(section[real_row], col)

            if real_col is None:
                continue

            value = parse_value(section[real_row][real_col])

            if index is None:
                return value

            seq = sequence_values(value)

            if seq is UNKNOWN:
                return UNKNOWN

            seq_index = sequence_index(index, len(seq))

            if seq_index is UNKNOWN:
                return UNKNOWN

            return seq[seq_index]

        return UNKNOWN

    def _adj_values(self, direction: str, indicator: str, rows: list[str] | None) -> list[Any] | object:
        neighbor = self.sections["neighbor"]

        real_direction = self._find_key(neighbor, direction)

        if real_direction is None:
            return UNKNOWN

        direction_state = neighbor[real_direction]

        if not isinstance(direction_state, dict):
            return UNKNOWN

        movement_rows = rows if rows is not None else [row for row in direction_state if norm_key(row) in {"et", "wt", "nt", "st", "el", "wl", "nl", "sl"}]

        indicator_norm = norm_key(indicator)

        values = []

        for row in movement_rows:
            real_row = self._find_key(direction_state, row)

            if real_row is None:
                return UNKNOWN

            queue, occupancy = self._parse_neighbor_pair(direction_state[real_row])

            if indicator_norm in {"queuelength", "queue"}:
                values.append(queue)

            elif indicator_norm in {"occupancy", "occ"}:
                values.append(occupancy)

            else:
                return UNKNOWN

        return values

    def _extreme(self, scope: str, indicator: str, op: str) -> dict[str, Any] | object:
        section_name = self._section_name(scope)

        if section_name is None:
            return UNKNOWN

        values = self._row_metric_values(section_name, indicator)

        if not values:
            return UNKNOWN

        reverse = norm_key(op) not in {"min"}

        sorted_values = sorted(values, key=lambda item: item[1], reverse=reverse)

        extreme_value = sorted_values[0][1]

        rows = [row for row, value in sorted_values if abs(value - extreme_value) <= 1e-6]

        return {"section": section_name, "rows": rows, "row": rows[0], "value": extreme_value}

    def _rank_rows(self, section_name: str, indicator: str) -> list[str] | object:
        values = self._row_metric_values(section_name, indicator)

        if not values:
            return UNKNOWN

        reverse = section_name != "simulation"

        return [row for row, _ in sorted(values, key=lambda item: item[1], reverse=reverse)]

    def _rank_groups(self, section_name: str, indicator: str) -> list[list[str]] | object:
        values = self._row_metric_values(section_name, indicator)

        if not values:
            return UNKNOWN

        reverse = section_name != "simulation"

        groups: list[list[str]] = []

        group_value: float | None = None

        for row, value in sorted(values, key=lambda item: item[1], reverse=reverse):
            if group_value is None or abs(value - group_value) > 1e-6:
                groups.append([row])

                group_value = value

            else:
                groups[-1].append(row)

        return groups

    def _row_metric_values(self, section_name: str, indicator: str) -> list[tuple[str, float]]:
        if section_name not in self.sections:
            return []

        rows = []

        for row, row_state in self.sections[section_name].items():
            if not isinstance(row_state, dict):
                continue

            value = self._resolve_cell(row, indicator)

            value_number = number(value)

            if value_number is not UNKNOWN:
                rows.append((row, value_number))

        return rows

    def _section_name(self, scope: str) -> str | None:
        scope_norm = norm_key(scope)

        if scope_norm in {"phase", "phases", "phaseaggregation"}:
            return "phase"

        if scope_norm in {"movement", "movements", "allmovements"}:
            return "movement"

        if scope_norm in {"simulation", "phaseresults", "phasesimulation", "simulated"}:
            return "simulation"

        return None

    def _trend(self, phase: str, indicator: str) -> str | object:
        history = self.sections["history"]

        real_phase = self._find_key(history, phase)

        if real_phase is None or not isinstance(history[real_phase], dict):
            return UNKNOWN

        phase_history = history[real_phase]

        summary_col = self._find_key(phase_history, "Trend Summary")

        if summary_col is not None:
            summary = norm_text(str(phase_history[summary_col]))

            if "increasing" in summary or "worsening" in summary:
                return "increasing"

            if "relieved" in summary or "decreasing" in summary:
                return "decreasing"

            if "stable" in summary:
                return "stable"

        real_indicator = self._find_key(phase_history, indicator)

        if real_indicator is None:
            return UNKNOWN

        values = [number(part) for part in str(phase_history[real_indicator]).replace("min", "").split("->")]

        if any(value is UNKNOWN for value in values) or len(values) < 2:
            return UNKNOWN

        return "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"

    def _history_value(self, phase: str, indicator: str) -> Any:
        history = self.sections["history"]

        real_phase = self._find_key(history, phase)

        if real_phase is None or not isinstance(history[real_phase], dict):
            return UNKNOWN

        real_indicator = self._find_key(history[real_phase], indicator)

        if real_indicator is None:
            return UNKNOWN

        return parse_value(history[real_phase][real_indicator])

    def _threshold_for(self, col: str, right_text: str) -> float | object:
        text = norm_text(right_text)

        col_norm = norm_key(col)

        number_match = re.search(r"[-+]?\d+(?:\.\d+)?", right_text)

        if number_match:
            return float(number_match.group(0))

        if "waiting" in col_norm:
            return 5.0

        if "occupancy" in col_norm:
            return 0.6 if "severe" in text else 0.4 if "congest" in text else 0.2

        if "downstreamqueue" in col_norm or "queue" in col_norm:
            return UNKNOWN

        return UNKNOWN

    def _cell_ref(self, text: str) -> dict[str, str | None]:
        parts = re.findall(r"\[([^\]]+)\]", text)

        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid cell reference: {text}")

        return {"row": parts[0].strip(), "col": parts[1].strip(), "index": parts[2].strip() if len(parts) == 3 else None}

    def _find_key(self, data: dict[str, Any], target: str) -> str | None:
        target_norm = norm_key(target)

        if target_norm.startswith("phase") and len(target_norm) > len("phase"):
            target_norm = target_norm[len("phase") :]

        for key in data:
            if norm_key(str(key)) == target_norm:
                return str(key)

        alias = ALIASES[target_norm] if target_norm in ALIASES else None

        if alias is None:
            return None

        for key in data:
            if norm_key(str(key)) in alias:
                return str(key)

        return None

    def _parse_neighbor_pair(self, value: Any) -> tuple[float, float]:
        parts = [part.strip() for part in str(value).split(",")]

        if len(parts) != 2:
            raise ValueError(f"Invalid neighbor pair: {value}")

        return float(parts[0]), float(parts[1])
