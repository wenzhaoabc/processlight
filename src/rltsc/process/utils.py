import pandas as pd


from .prompts import PROMPTS


def fnumber(x: float | int, dig=1) -> str:
    if x == 0.0 or x == 0:
        return "0"

    res = f"{x:.{dig}f}"

    while res.endswith("0"):
        res = res[:-1]

    if res.endswith("."):
        res = res[:-1]

    return res


def table_to_dict(rows: list[list], headers: list[str], index_col: str) -> dict[str, dict[str, str]]:
    df = pd.DataFrame(rows, columns=headers)

    df.set_index(index_col, inplace=True)

    result_dict = df.to_dict("index")

    return result_dict


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    table = [headers, ["---"] * len(headers), *rows]

    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in table)


def _dict_table_to_rows(table_dict: dict[str, dict[str, object]], headers: list[str], index_col: str) -> list[list[object]]:
    rows: list[list[object]] = []

    for row_key, row_dict in table_dict.items():
        rows.append([row_key, *[row_dict[header] for header in headers if header != index_col]])

    return rows


def build_process_prompt_text(prompt_state_dict: dict[str, dict], phase_list: list[str]) -> str:
    movement_headers = [
        "Movement",
        "Phase",
        "Early Queue Length",
        "Downstream Occupancy",
        "Pressure",
        "Avg Waiting Time(minutes)",
        "Segment Veh Occupancy",
        "Vehicle Count",
    ]

    phase_headers = [
        "Phase",
        "Movement",
        "Early Queue Length",
        "Pressure",
        "Avg Waiting Time(minutes)",
        "Segment Veh Count",
        "Vehicle Count",
    ]

    neighbor_movements: list[str] = []

    for phase in phase_list:
        neighbor_movements.extend([phase[:2], phase[-2:]])

    neighbor_headers = ["Direction", "Intersection ID", *neighbor_movements, "Related Movements"]

    history_headers = ["Phase", "Queue Change", "Pressure Change", "Waiting Change", "Service Gap", "Trend Summary"]

    movement_rows = _dict_table_to_rows(prompt_state_dict["movement"], movement_headers, index_col="Movement")

    phase_rows = _dict_table_to_rows(prompt_state_dict["phase"], phase_headers, index_col="Phase")

    neighbor_rows = _dict_table_to_rows(prompt_state_dict["neighbor"], neighbor_headers, index_col="Direction")

    history_phase_dict = {phase: phase_history for phase, phase_history in prompt_state_dict["history"].items() if isinstance(phase_history, dict)}

    history_rows = _dict_table_to_rows(history_phase_dict, history_headers, index_col="Phase")

    movement_table = _md_table(movement_headers, movement_rows)

    phase_table = _md_table(phase_headers, phase_rows)

    neighbor_table = _md_table(neighbor_headers, neighbor_rows)

    phase_history_table = _md_table(history_headers, history_rows)

    return "\n".join(
        [
            PROMPTS["BACKGROUND"].format(LEGAL_PHASES=", ".join(phase_list)),
            PROMPTS["RULES"],
            PROMPTS["LOCAL"].format(
                LOCAL_STATE_TABLE="\n\n".join(
                    [
                        "### Movement State",
                        movement_table,
                        "### Phase Aggregation",
                        phase_table,
                    ]
                )
            ),
            PROMPTS["NEIGHBOR"].format(
                NEIGHBOR_STATE_TABLE=neighbor_table,
            ),
            PROMPTS["HISTORY"].format(
                HISTORY_WINDOW=prompt_state_dict["history"]["history_window"],
                RECENT_PHASES=prompt_state_dict["history"]["recent_phases"],
                LAST_PHASE=prompt_state_dict["history"]["last_phase"],
                PHASE_HISTORY_TABLE=phase_history_table,
            ),
            PROMPTS["FORMAT"],
        ]
    )
