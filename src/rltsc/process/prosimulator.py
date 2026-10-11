from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiofiles
from openai import AsyncOpenAI

from rltsc.atlm.async_cityflow_env import AsyncAction, AsyncCityFlowEnv
from rltsc.atlm.configs import CityFlowConfig, LLMConfig, SimulatorConfig
from rltsc.atlm.snapshot_io import dump_cityflow_snapshot

from .prompts import PROMPTS
from .utils import fnumber, table_to_dict


class ProSimulateWrapper:
    def __init__(
        self,
        cityflow_config: CityFlowConfig,
        simulator_config: SimulatorConfig,
        traj_name: str,
        llm_config: LLMConfig,
        short_id: str | None = None,
        local_inter_id: str | None = None,
        save_snapshot: bool = False,
    ):
        self.simulator_config = simulator_config

        self.cityflow_config = cityflow_config

        self.work_dir = Path(cityflow_config.dir)

        self.snapshot_lock_path = self.work_dir.parent / ".cityflow_snapshot.lock"

        self.traj_name = traj_name

        self.short_id = short_id or traj_name[:12]

        self.llm_config = llm_config

        self.save_snapshot = save_snapshot

        self._setup_logger()

        if simulator_config.phase_list_type == "four":
            self.phases_id = {"ETWT": 1, "NTST": 2, "ELWL": 3, "NLSL": 4}

        else:
            self.phases_id = {"ETWT": 1, "NTST": 2, "ELWL": 3, "NLSL": 4, "ELET": 5, "WLWT": 6, "NLNT": 7, "SLST": 8}

        self.phase_list = list(self.phases_id.keys())

        self.id_phases = {value: key for key, value in self.phases_id.items()}

        self._prepare_env_files()

        self.env = AsyncCityFlowEnv(
            config=cityflow_config,
            yellow_phase=simulator_config.yellow_phase_id,
            yellow_duration=simulator_config.yellow_duration,
            always_insert_yellow=simulator_config.always_insert_yellow,
            duration_includes_yellow=simulator_config.duration_includes_yellow,
            phase_list_type=simulator_config.phase_list_type,
            max_simulation_seconds=simulator_config.max_simulation_seconds,
            local_inter_id=local_inter_id,
        )

        self.env.reset()

        self._client = AsyncOpenAI(base_url=llm_config.llm_base_url, api_key=llm_config.llm_api_key)

        self.phase_duration = simulator_config.fixed_phase_duration

        params = simulator_config.strategy_params

        self.per_vehicle_road_occupation = float(params["density_capacity_vehicle_length"]) if "density_capacity_vehicle_length" in params else 9.0

        self.llm_max_retries = int(params["llm_max_retries"]) if "llm_max_retries" in params else 3

        self.history_window = int(params["history_window"]) if "history_window" in params else 4

        self.lane_vehicles: dict[str, list[str]]

        self.lane_waiting_vehicle_count: dict[str, int]

        self.instant_inter_state: dict[str, dict[str, Any]] = {}

        self.history_inter_state: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self.decision_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self.snapshot_stats = {
            "attempted": 0,
            "saved": 0,
            "failed": 0,
        }

    def _setup_logger(self):
        self.logger = logging.getLogger(f"processlight_{self.short_id}")

        self.logger.setLevel(logging.INFO)

        self.logger.handlers.clear()

        log_path = self.work_dir / "processlight.log"

        file_handler = logging.FileHandler(log_path)

        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger.addHandler(file_handler)

    def _prepare_env_files(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.cityflow_config.roadnetFile, self.work_dir)

        shutil.copy(self.cityflow_config.flowFile, self.work_dir)

        self.roadnet_file = Path(self.cityflow_config.roadnetFile).name

        self.traffic_file = Path(self.cityflow_config.flowFile).name

        self.cityflow_config.roadnetFile = self.roadnet_file

        self.cityflow_config.flowFile = self.traffic_file

    async def run_simulation(self) -> tuple[dict, list]:
        control_mode = "process-simulation"

        self.logger.info("Starting simulation: %s (%s), control=%s", self.short_id, self.traj_name, control_mode)

        total_run_time = self.simulator_config.max_simulation_seconds

        done = False

        self.env.reset()

        metrics = {"per_step_reward": []}

        current_time = self.env.get_current_time()

        simulate_start_time = time.time()

        while not done and current_time < total_run_time:
            ready_inter_ids = self.env.find_ready_intersections()

            self._update_lane_raw_state()

            self.instant_inter_state = self._inters_instant_state(ready_inter_ids)

            tasks = [self.async_decide_inter_tl(inter_id) for inter_id in ready_inter_ids]

            results = await asyncio.gather(*tasks)

            if self.save_snapshot:
                snapshot_saved = self._save_snapshot_bundle(current_time, ready_inter_ids, ready_inter_ids)

                if not snapshot_saved:
                    self.logger.error("Failed to save snapshot for time: %d", current_time)

            final_action_list = {}

            for inter_id, phase in zip(ready_inter_ids, results, strict=False):
                final_action_list[inter_id] = AsyncAction(
                    phase_id=self.phases_id[phase],
                    duration=float(int(float(self.phase_duration))),
                )

            _, reward, done, _ = self.env.step(final_action_list)

            current_time = self.env.get_current_time()

            metrics["per_step_reward"].append(sum(reward.values()))

            if current_time % 300 <= 20:
                self.logger.info("Current simulation time: %d seconds", current_time)

                await self.save_simulation_logs()

        await self.save_simulation_logs()

        simulate_end_time = time.time()

        env_metrics = self.env.get_final_metrics()

        simulation_result = {
            "simulate_time": simulate_end_time - simulate_start_time,
            "total_reward": float(sum(metrics["per_step_reward"])),
            **env_metrics,
            "avg_travel_time": self.env.eng.get_average_travel_time(),
            "cityflow_att": self.env.eng.get_average_travel_time(),
            "traffic_file": self.traffic_file,
            "roadnet_file": self.roadnet_file,
            "traj_name": self.traj_name,
        }

        simulation_result["snapshot_stats"] = dict(self.snapshot_stats)

        await self.async_save_file(self.work_dir / "simulation_result.json", json.dumps(simulation_result, indent=2, ensure_ascii=False))

        return simulation_result, []

    def _update_lane_raw_state(self):
        self.lane_vehicles = self.env.eng.get_lane_vehicles()

        self.lane_waiting_vehicle_count = self.env.eng.get_lane_waiting_vehicle_count()

    def _inters_instant_state(self, inter_ids: list[str]):
        inter_states = {}

        for inter_id in inter_ids:
            inter_states[inter_id] = self._get_inter_instant_state(inter_id)

        return inter_states

    def _get_inter_instant_state(self, inter_id: str):
        inter_movement_lanes = self.env.intersections_phases_lanes[inter_id]

        movement_states = {}

        for movement, lane_info in inter_movement_lanes.items():
            queue = 0

            segment_vehicle_count = [0, 0, 0]

            segment_vehicle_occupancy = [0.0, 0.0, 0.0]

            vehicle_count = 0

            from_lane = lane_info["from"][0]

            lane_length = self.env.lane_properties[from_lane]["length"]

            queue = self.lane_waiting_vehicle_count[from_lane]

            vehicles = self.lane_vehicles[from_lane]

            vehicle_count = len(vehicles)

            for vehicle_id in vehicles:
                vehicle_info = self.env.eng.get_vehicle_info(vehicle_id)

                speed = float(vehicle_info["speed"])

                distance = float(vehicle_info["distance"])

                lane_pos = lane_length - distance

                if speed > 0.1:
                    if lane_pos < lane_length / 10:
                        segment_vehicle_count[0] += 1

                    elif lane_pos < lane_length / 3:
                        segment_vehicle_count[1] += 1

                    else:
                        segment_vehicle_count[2] += 1

            segment_vehicle_occupancy[0] = segment_vehicle_count[0] * self.per_vehicle_road_occupation / max(lane_length / 10, 1.0)

            segment_vehicle_occupancy[1] = segment_vehicle_count[1] * self.per_vehicle_road_occupation / max(lane_length / 3 - lane_length / 10, 1.0)

            segment_vehicle_occupancy[2] = segment_vehicle_count[2] * self.per_vehicle_road_occupation / max(lane_length - lane_length / 3, 1.0)

            avg_waiting_time = 0.0

            max_waiting_time = 0.0

            if vehicles:
                lane_waiting_time = self.env.lane_vehicle_waiting_time[from_lane]

                vehicle_waiting_times = [lane_waiting_time[vehicle_id] if vehicle_id in lane_waiting_time else 0.0 for vehicle_id in vehicles]

                avg_waiting_time = sum(vehicle_waiting_times) / len(vehicles)

                max_waiting_time = max(vehicle_waiting_times)

            downstream_lanes = [lane for lane in lane_info["to"]]

            downstream_queue = sum([self.lane_waiting_vehicle_count[lane] for lane in downstream_lanes])

            downstream_avg_queue = downstream_queue / len(downstream_lanes) if downstream_lanes else 0

            downstream_avg_vehicle_count = sum([len(self.lane_vehicles[lane]) for lane in downstream_lanes]) / len(downstream_lanes) if downstream_lanes else 0

            upstream_lanes = []

            lane_start_inter = self.env.lane_properties[from_lane]["start_end_inter"][0]

            if lane_start_inter in self.env.intersections_phases_lanes:
                for upstream_movement_lanes in self.env.intersections_phases_lanes[lane_start_inter].values():
                    if from_lane in upstream_movement_lanes["to"]:
                        upstream_lanes.extend(upstream_movement_lanes["from"])

            upstream_queue = sum([self.lane_waiting_vehicle_count[lane] for lane in upstream_lanes])

            movement_states[movement] = {
                "movement": movement,
                "phase": movement[:2] + movement[-2:],
                "queue": queue,
                "vehicle_count": vehicle_count,
                "segment_vehicle_count": segment_vehicle_count,
                "segment_vehicle_occupancy": segment_vehicle_occupancy,
                "avg_waiting_time": avg_waiting_time,
                "max_waiting_time": max_waiting_time,
                "downstream_queue": downstream_queue,
                "downstream_avg_queue": downstream_avg_queue,
                "downstream_avg_vehicle_count": downstream_avg_vehicle_count,
                "downstream_occupancy": downstream_avg_vehicle_count * self.per_vehicle_road_occupation / max(lane_length, 1.0),
                "pressure": queue - downstream_avg_queue,
                "occupancy": vehicle_count * self.per_vehicle_road_occupation / max(lane_length, 1.0),
                "upstream_queue": upstream_queue,
                "upstream_lanes": upstream_lanes,
            }

        return movement_states

    def _md_table(self, headers: list[str], rows: list[list[Any]]) -> str:
        table = [headers, ["---"] * len(headers), *rows]

        return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in table)

    def _format_segment(self, segment_vehicle_count: list[int | float], dig: int) -> str:
        if not segment_vehicle_count:
            return "[]"

        if isinstance(segment_vehicle_count[0], float):
            return "[" + ", ".join(fnumber(x, dig) for x in segment_vehicle_count) + "]"

        return "[" + ", ".join(map(str, segment_vehicle_count)) + "]"

    def _direction_neighbor_state(self, inter_id: str) -> dict[str, list[str]]:
        direction_states = {}

        for direction in ["East", "West", "North", "South"]:
            upstream_to_lanes = []

            mv1, mv2 = direction[0] + "T", direction[0] + "L"

            upstream_to_lanes = [self.env.intersections_phases_lanes[inter_id][mv1]["from"][0], self.env.intersections_phases_lanes[inter_id][mv2]["from"][0]]

            direction_inter = "_".join(["intersection"] + upstream_to_lanes[0].split("_")[1:3])

            to_current_lane_mvs = []

            if direction_inter in self.env.intersections_phases_lanes:
                for mv, mv_lanes in self.env.intersections_phases_lanes[direction_inter].items():
                    if upstream_to_lanes[0] in mv_lanes["to"] or upstream_to_lanes[1] in mv_lanes["to"]:
                        if mv not in to_current_lane_mvs:
                            to_current_lane_mvs.append(mv)

                dir_inter_state = self.instant_inter_state[direction_inter]

                direction_states[direction] = {mv: {"queue": dir_inter_state[mv]["queue"], "occupancy": dir_inter_state[mv]["occupancy"]} for mv in dir_inter_state}

                direction_states[direction]["mvs"] = ",".join(to_current_lane_mvs) + " -> " + ",".join([mv1, mv2])

                direction_states[direction]["intersection_id"] = direction_inter

        return direction_states

    def _build_intersection_prompt(self, inter_id: str, phase_state: dict[str, dict[str, Any]], neighbor_state: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        prompt_state_dict: dict[str, Any] = {}

        movement_phase = {}

        for phase in self.phase_list:
            movement_phase[phase[:2]] = phase

            movement_phase[phase[-2:]] = phase

        movement_rows = []

        preferred_order = ["ET", "WT", "NT", "ST", "EL", "WL", "NL", "SL"]

        movement_order = [movement for movement in preferred_order if movement in self.instant_inter_state[inter_id]]

        movement_order.extend(sorted([movement for movement in self.instant_inter_state[inter_id] if movement not in preferred_order]))

        for movement in movement_order:
            state = self.instant_inter_state[inter_id][movement]

            movement_rows.append(
                [
                    movement,
                    movement_phase[movement],
                    state["queue"],
                    state["downstream_occupancy"],
                    fnumber(state["pressure"], 1),
                    fnumber(state["avg_waiting_time"] / 60, 1),
                    self._format_segment(state["segment_vehicle_occupancy"], 1),
                    state["vehicle_count"],
                ]
            )

        movement_headers = ["Movement", "Phase", "Early Queue Length", "Downstream Occupancy", "Pressure", "Avg Waiting Time(minutes)", "Segment Veh Occupancy", "Vehicle Count"]

        movement_table = self._md_table(
            movement_headers,
            movement_rows,
        )

        prompt_state_dict["movement"] = table_to_dict(movement_rows, movement_headers, index_col="Movement")

        phase_rows = []

        for phase in self.phase_list:
            state = phase_state[phase]

            phase_rows.append(
                [
                    phase,
                    state["movements"],
                    state["queue"],
                    state["downstream_queue"],
                    fnumber(state["pressure"], 1),
                    fnumber(state["avg_waiting_time"] / 60, 1),
                    self._format_segment(state["segment_vehicle_count"], 1),
                    state["vehicle_count"],
                ]
            )

        phase_headers = ["Phase", "Movement", "Early Queue Length", "Downstream Queue Length", "Pressure", "Avg Waiting Time(minutes)", "Segment Veh Count", "Vehicle Count"]

        phase_table = self._md_table(
            phase_headers,
            phase_rows,
        )

        prompt_state_dict["phase"] = table_to_dict(phase_rows, phase_headers, index_col="Phase")

        neighbor_rows = []

        neighbor_movements = []

        for phase in self.phase_list:
            neighbor_movements.extend([phase[:2], phase[-2:]])

        for direction in ["East", "West", "North", "South"]:
            if direction in neighbor_state:
                neighbor_rows.append(
                    [
                        direction,
                        neighbor_state[direction]["intersection_id"],
                        *[f"{neighbor_state[direction][mv]['queue']}, {fnumber(neighbor_state[direction][mv]['occupancy'], 1)}" for mv in neighbor_movements],
                        neighbor_state[direction]["mvs"],
                    ]
                )

        neighbor_headers = ["Direction", "Intersection ID", *neighbor_movements, "Related Movements"]

        neighbor_table = self._md_table(
            neighbor_headers,
            neighbor_rows,
        )

        prompt_state_dict["neighbor"] = table_to_dict(neighbor_rows, neighbor_headers, index_col="Direction")

        completed_logs = [item for item in self.decision_logs[inter_id] if "phase" in item]

        history_logs = completed_logs[-(self.history_window - 1) :] if self.history_window > 1 else []

        phase_history_rows = []

        for phase in self.phase_list:
            historical_phase_states = [item["phase_state"][phase] for item in history_logs]

            historical_phase_states.append(phase_state[phase])

            queue_values = [float(item["queue"]) for item in historical_phase_states]

            pressure_values = [float(item["pressure"]) for item in historical_phase_states]

            waiting_values = [float(item["avg_waiting_time"]) / 60 for item in historical_phase_states]

            service_gap = None

            for steps_ago, item in enumerate(reversed(completed_logs)):
                if item["phase"] == phase:
                    service_gap = steps_ago

                    break

            service_gap_text = "not served recently" if service_gap is None else f"{service_gap} steps"

            if len(queue_values) < 2:
                trend_summary = "current state only"

            else:
                queue_delta = queue_values[-1] - queue_values[0]

                waiting_delta = waiting_values[-1] - waiting_values[0]

                long_unserved = service_gap is None or service_gap >= self.history_window

                if long_unserved and (queue_delta > 0 or waiting_delta > 0):
                    trend_summary = "unserved for long and demand is worsening"

                elif queue_delta >= 3 and waiting_delta >= 0.5:
                    trend_summary = "demand is increasing and waiting is accumulating"

                elif queue_delta >= 3:
                    trend_summary = "demand is increasing"

                elif queue_delta <= -3 or waiting_delta <= -0.5:
                    trend_summary = "recently served and clearly relieved"

                else:
                    trend_summary = "stable"

            phase_history_rows.append(
                [
                    phase,
                    " -> ".join(fnumber(value) for value in queue_values),
                    " -> ".join(fnumber(value) for value in pressure_values),
                    " -> ".join(fnumber(value) for value in waiting_values) + " min",
                    service_gap_text,
                    trend_summary,
                ]
            )

        phase_history_headers = ["Phase", "Queue Change", "Pressure Change", "Waiting Change", "Service Gap", "Trend Summary"]

        phase_history_table = self._md_table(
            phase_history_headers,
            phase_history_rows,
        )

        prompt_state_dict["history"] = table_to_dict(phase_history_rows, phase_history_headers, index_col="Phase")

        recent_logs = completed_logs[-self.history_window :]

        recent_phases = "none" if not recent_logs else " -> ".join([item["phase"] for item in recent_logs])

        last_phase = self._get_runtime_info(inter_id)["current_phase"] if not completed_logs else completed_logs[-1]["phase"]

        prompt_state_dict["history"]["history_window"] = self.history_window

        prompt_state_dict["history"]["recent_phases"] = recent_phases

        prompt_state_dict["history"]["last_phase"] = last_phase

        res_prompt = "\n".join(
            [
                PROMPTS["BACKGROUND"].format(LEGAL_PHASES=", ".join(self.phase_list)),
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
                    HISTORY_WINDOW=self.history_window,
                    RECENT_PHASES=recent_phases,
                    LAST_PHASE=last_phase,
                    PHASE_HISTORY_TABLE=phase_history_table,
                ),
                PROMPTS["FORMAT"],
            ]
        )

        return res_prompt, prompt_state_dict

    async def async_decide_inter_tl(self, inter_id):
        inter_state = self.instant_inter_state[inter_id]

        phase_state = {}

        for phase in self.phase_list:
            mv1, mv2 = phase[:2], phase[-2:]

            state1 = inter_state[mv1]

            state2 = inter_state[mv2]

            vehicle_count = state1["vehicle_count"] + state2["vehicle_count"]

            downstream_avg_vehicle_count = state1["downstream_avg_vehicle_count"] + state2["downstream_avg_vehicle_count"]

            downstream_occupancy = 0.0

            if downstream_avg_vehicle_count > 0:
                downstream_occupancy = (
                    state1["downstream_occupancy"] * state1["downstream_avg_vehicle_count"] + state2["downstream_occupancy"] * state2["downstream_avg_vehicle_count"]
                ) / downstream_avg_vehicle_count

            phase_state[phase] = {
                "queue": state1["queue"] + state2["queue"],
                "vehicle_count": vehicle_count,
                "pressure": state1["pressure"] + state2["pressure"],
                "downstream_queue": state1["downstream_queue"] + state2["downstream_queue"],
                "upstream_queue": state1["upstream_queue"] + state2["upstream_queue"],
                "movements": ",".join([mv1, mv2]),
                "segment_vehicle_count": [state1["segment_vehicle_count"][i] + state2["segment_vehicle_count"][i] for i in range(3)],
                "segment_vehicle_occupancy": [(state1["segment_vehicle_occupancy"][i] + state2["segment_vehicle_occupancy"][i]) / 2 for i in range(3)],
                "downstream_occupancy": downstream_occupancy,
                "max_waiting_time": max(state1["max_waiting_time"], state2["max_waiting_time"]),
                "avg_waiting_time": (state1["avg_waiting_time"] * state1["vehicle_count"] + state2["avg_waiting_time"] * state2["vehicle_count"]) / max(vehicle_count, 1),
            }

        neighbor_state = self._direction_neighbor_state(inter_id)

        self.decision_logs[inter_id].append(
            {
                "timestamp": self.env.get_current_time(),
                "state": inter_state,
                "phase_state": phase_state,
                "neighbor_state": neighbor_state,
            }
        )

        inter_vehicle_count = sum([inter_state[movement]["vehicle_count"] for movement in inter_state])

        if inter_vehicle_count == 0:
            self.decision_logs[inter_id][-1].update(
                {
                    "llm_call_log": {"reason": "no vehicles"},
                    "phase": self.phase_list[0],
                }
            )

            return self.phase_list[0]

        prompt, prompt_state_dict = self._build_intersection_prompt(inter_id, phase_state, neighbor_state)

        response_text, call_log = await self._call_llm_retry(prompt)

        try:
            phase = self._parse_phase_from_response(response_text)

        except Exception as exc:
            self.logger.error("Failed to parse LLM response for intersection %s: %s", inter_id, str(exc))

            phase = self.phase_list[0]

        self.decision_logs[inter_id][-1].update(
            {
                "prompt_state_dict": prompt_state_dict,
                "llm_call_log": call_log,
                "phase": phase,
            }
        )

        return phase

    async def _call_llm_retry(self, prompt: str) -> tuple[str, dict[str, Any]]:
        messages = [
            {"role": "system", "content": PROMPTS["SYSTEM"]},
            {"role": "user", "content": prompt},
        ]

        params = {"model": self.llm_config.llm_model_name, "extra_body": {"thinking": {"type": "disabled"}, "enable_thinking": False}}

        call_log: dict[str, Any] = {"llm_parameters": params, "messages": list(messages)}

        for attempt in range(self.llm_max_retries):
            try:
                response = await self._client.chat.completions.create(messages=messages, **params)

                response_text = response.choices[0].message.content or ""

                if not response_text.strip():
                    raise ValueError("Empty response from LLM")

                call_log["messages"].append({"role": "assistant", "content": response_text})

                phase = self._parse_phase_from_response(response_text)

                if phase not in self.phase_list:
                    raise ValueError(f"Invalid phase '{phase}' extracted from LLM response")

                return response_text, call_log

            except Exception as exc:
                call_log["last_error"] = str(exc)

                if attempt == self.llm_max_retries - 1:
                    self.logger.error("LLM call failed after %d attempts: %s", self.llm_max_retries, str(exc))

        return f"<phase>{self.phase_list[0]}</phase>", call_log

    def _get_runtime_info(self, intersection_id: str) -> dict[str, Any]:
        state = self.env.get_intersection_tl_state(intersection_id)

        if state.current_phase in self.id_phases:
            current_phase = self.id_phases[state.current_phase]

        else:
            current_phase = str(state.current_phase)

        return {
            "current_phase": current_phase,
            "current_phase_dur": int(self.env.get_current_time() - state.last_action_time),
        }

    def _parse_phase_from_response(self, response_text: str) -> str:
        match = re.findall(r"<phase>(.*?)</phase>", response_text, re.IGNORECASE | re.DOTALL)

        if match:
            return match[-1].strip()

        else:
            raise ValueError("No phase found in LLM response")

    async def async_save_file(self, file_path: str, content: str | bytes):
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

    async def save_simulation_logs(self):
        log_dir = self.work_dir / "action_logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        tasks = []

        for inter_id, inter_log in self.decision_logs.items():
            log_file = log_dir / f"{inter_id}_action_log.json"

            tasks.append(self.async_save_file(log_file.as_posix(), json.dumps(inter_log, indent=2, ensure_ascii=False)))

        await asyncio.gather(*tasks)

    def _save_snapshot_bundle(self, current_time: float, ready_intersections: list[str], selected_intersections: list[str]) -> bool:
        snapshot_folder = self.work_dir / "archives" / f"{int(current_time)}"

        snapshot_folder.mkdir(parents=True, exist_ok=True)

        snapshot_saved = dump_cityflow_snapshot(
            engine=self.env.eng,
            snapshot_path=snapshot_folder / "snapshot.json",
            lock_path=self.snapshot_lock_path,
            logger=self.logger,
            current_time=current_time,
        )

        self.snapshot_stats["attempted"] += 1

        self.snapshot_stats["saved"] += int(snapshot_saved)

        self.snapshot_stats["failed"] += int(not snapshot_saved)

        with open(snapshot_folder / "inter_states.json", "w") as f:
            json.dump([asdict(k) for k in self.env.inters_state], f)

        with open(snapshot_folder / "ready_intersections.json", "w") as f:
            json.dump(ready_intersections, f)

        with open(snapshot_folder / "selected_intersections.json", "w") as f:
            json.dump(selected_intersections if snapshot_saved else [], f)

        with open(snapshot_folder / "snapshot_status.json", "w") as f:
            json.dump(
                {
                    "snapshot_saved": snapshot_saved,
                    "current_time": current_time,
                    "selected_intersections": selected_intersections if snapshot_saved else [],
                },
                f,
            )

        return snapshot_saved
