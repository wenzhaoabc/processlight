import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import cityflow

from rltsc.atlm.configs import CityFlowConfig, VehicleInfo


def calculate_road_length(road_points):
    length = 0.0

    i = 1

    while i < len(road_points):
        length += math.sqrt((road_points[i]["x"] - road_points[i - 1]["x"]) ** 2 + (road_points[i]["y"] - road_points[i - 1]["y"]) ** 2)

        i += 1

    return length


def calculate_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


class CityFlowEngine:
    def __init__(self, config: CityFlowConfig):
        if not config.dir.endswith("/"):
            config.dir = config.dir + "/"

        self.config = config

        self._write_config_file()

        self.eng = cityflow.Engine(os.path.join(self.config.dir, "cityflow_config.json"), thread_num=8)

    def _write_config_file(self):
        config_dict = self.config.__dict__

        with open(os.path.join(self.config.dir, "cityflow_config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    def get_vehicle_count(self) -> int:
        return self.eng.get_vehicle_count()

    def get_vehicles(self, include_waiting: bool = False) -> list[str]:
        return self.eng.get_vehicles(include_waiting=include_waiting)

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return self.eng.get_lane_vehicle_count()

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        return self.eng.get_lane_waiting_vehicle_count()

    def get_lane_vehicles(self) -> dict[str, list[str]]:
        return self.eng.get_lane_vehicles()

    def get_vehicle_info(self, vehicle_id: str) -> VehicleInfo:
        v_info = self.eng.get_vehicle_info(vehicle_id)

        return VehicleInfo(
            running=bool(v_info["running"]),
            speed=float(v_info.get("speed", 0.0)),
            distance=float(v_info.get("distance", 0.0)),
            drivable=v_info.get("drivable", ""),
            road=v_info.get("road", ""),
            intersection=v_info.get("intersection", ""),
            route=v_info.get("route", ""),
        )

    def get_vehicle_speed(self) -> dict[str, float]:
        return self.eng.get_vehicle_speed()

    def get_vehicle_distance(self) -> dict[str, float]:
        return self.eng.get_vehicle_distance()

    def get_leader(self, vehicle_id: str) -> str:
        return self.eng.get_leader(vehicle_id)

    def get_current_time(self) -> float:
        return self.eng.get_current_time()

    def get_average_travel_time(self) -> float:
        return self.eng.get_average_travel_time()

    def set_tl_phase(self, intersection_id: str, phase_id: int):
        self.eng.set_tl_phase(intersection_id, phase_id)

    def set_vehicle_speed(self, vehicle_id: str, speed: float):
        self.eng.set_vehicle_speed(vehicle_id, speed)

    def reset(self, seed: bool = False):
        self.eng.reset(seed=seed)

    def set_random_seed(self, seed: int):
        self.eng.set_random_seed(seed)

    def next_step(self):
        self.eng.next_step()

    def snapshot(self) -> "cityflow.Archive":
        return self.eng.snapshot()

    def load_from_file(self, snapshot_file: str):
        self.eng.load_from_file(snapshot_file)


IntersectionID = str


PhaseID = int


Seconds = float


ObsFunc = Callable[[CityFlowEngine, IntersectionID], Any]


RewardFunc = Callable[[CityFlowEngine, IntersectionID], Any]


@dataclass
class IntersectionState:
    intersection_id: IntersectionID

    current_phase: PhaseID

    next_action_time: Seconds = 0.0

    last_action_time: Seconds = 0.0

    is_yellow: bool = False

    yellow_end_time: Seconds = 0.0

    min_green: Seconds = 0.0

    def to_dict(self):
        return {
            "intersection_id": self.intersection_id,
            "current_phase": self.current_phase,
            "next_action_time": self.next_action_time,
            "last_action_time": self.last_action_time,
            "is_yellow": self.is_yellow,
            "yellow_end_time": self.yellow_end_time,
            "min_green": self.min_green,
        }


@dataclass
class AsyncAction:
    phase_id: PhaseID

    duration: Seconds


class MetricsTracker:
    def __init__(
        self,
        roadnet: dict[str, Any],
        eng: CityFlowEngine,
        inter_id: str | None = None,
        incoming_lanes: list[str] | None = None,
        outgoing_lanes: list[str] | None = None,
    ):
        self.roadnet = roadnet

        self.eng = eng

        self.veh_waiting_time: dict[str, dict] = {}

        self.vehicles = []

        self.waiting_time = 0.0

        self.all_seen_vehicles = set()

        self.last_active_vehicles: set[str] = set()

        self.vehicle_enter_time: dict[str, float] = {}

        self.vehicle_leave_time: dict[str, float] = {}

        self.completed_travel_times: dict[str, float] = {}

        self.last_vehicle_speeds: dict[str, float] = {}

        self.emergency_braking_count: int = 0

        self.emergency_braking_vehicles: set = set()

        self.braking_threshold: float = 3.0

        self.total_queue_std = 0.0

        self.step_count = 0

        self.total_avg_speed = 0.0

        self.local_seen_vehicles = set()

        self.local_waiting_time = 0.0

        self.local_incoming_wt = 0.0

        self.local_outgoing_wt = 0.0

        self.local_ebc: int = 0

        self.local_emergency_braking_vehicles: set[str] = set()

        self.local_incoming_ebc: int = 0

        self.local_outgoing_ebc: int = 0

        self.local_inter_id = inter_id

        self.local_incoming_lanes = incoming_lanes or []

        self.local_outgoing_lanes = outgoing_lanes or []

    def reset(self):
        self.veh_waiting_time = {}

        self.vehicles = []

        self.waiting_time = 0.0

        self.all_seen_vehicles = set()

        self.last_active_vehicles = set()

        self.vehicle_enter_time = {}

        self.vehicle_leave_time = {}

        self.completed_travel_times = {}

        self.last_vehicle_speeds = {}

        self.emergency_braking_count = 0

        self.emergency_braking_vehicles = set()

        self.total_queue_std = 0.0

        self.step_count = 0

        self.total_avg_speed = 0.0

        self.local_seen_vehicles = set()

        self.local_waiting_time = 0.0

        self.local_incoming_wt = 0.0

        self.local_outgoing_wt = 0.0

        self.local_ebc = 0

        self.local_emergency_braking_vehicles = set()

        self.local_incoming_ebc = 0

        self.local_outgoing_ebc = 0

    def update(self):
        current_time = self.eng.get_current_time()

        vehicles = self.eng.get_vehicles(include_waiting=True)

        active_vehicles = set(vehicles)

        self.all_seen_vehicles.update(active_vehicles)

        self.vehicles = list(set(self.vehicles) | active_vehicles)

        for v_id in active_vehicles:
            if v_id not in self.vehicle_enter_time:
                self.vehicle_enter_time[v_id] = current_time

        left_vehicles = self.last_active_vehicles - active_vehicles

        for v_id in left_vehicles:
            if v_id in self.vehicle_enter_time and v_id not in self.completed_travel_times:
                self.vehicle_leave_time[v_id] = current_time

                self.completed_travel_times[v_id] = current_time - self.vehicle_enter_time[v_id]

        self.last_active_vehicles = active_vehicles

        all_lane_vehicles = self.eng.get_lane_vehicles()

        local_incoming_vehicles = list(dict.fromkeys(chain.from_iterable(all_lane_vehicles.get(lane, []) for lane in self.local_incoming_lanes)))

        local_outgoing_vehicles = list(dict.fromkeys(chain.from_iterable(all_lane_vehicles.get(lane, []) for lane in self.local_outgoing_lanes)))

        self.local_seen_vehicles.update(local_incoming_vehicles + local_outgoing_vehicles)

        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()

        self.waiting_time += sum(lane_waiting_count.values()) * 1.0

        for lane in self.local_incoming_lanes:
            if lane in lane_waiting_count:
                self.local_incoming_wt += lane_waiting_count[lane] * 1.0

        for lane in self.local_outgoing_lanes:
            if lane in lane_waiting_count:
                self.local_outgoing_wt += lane_waiting_count[lane] * 1.0

        self.local_waiting_time = self.local_incoming_wt + self.local_outgoing_wt

        queues = list(lane_waiting_count.values())

        if queues:
            mean_q = sum(queues) / len(queues)

            variance = sum((x - mean_q) ** 2 for x in queues) / len(queues)

            std_q = math.sqrt(variance)

            self.total_queue_std += std_q

        self.step_count += 1

        current_speeds = self.eng.get_vehicle_speed()

        if current_speeds:
            avg_speed_step = sum(current_speeds.values()) / len(current_speeds)

            self.total_avg_speed += avg_speed_step

        interval = self.eng.config.interval if hasattr(self.eng, "config") else 1.0

        for v_id, current_speed in current_speeds.items():
            if v_id in self.last_vehicle_speeds:
                prev_speed = self.last_vehicle_speeds[v_id]

                acceleration = (current_speed - prev_speed) / interval

                if acceleration < -self.braking_threshold:
                    self.emergency_braking_count += 1

                    self.emergency_braking_vehicles.add(v_id)

                    if v_id in local_incoming_vehicles:
                        self.local_incoming_ebc += 1

                        self.local_ebc += 1

                    if v_id in local_outgoing_vehicles:
                        self.local_outgoing_ebc += 1

                        self.local_ebc += 1

        self.last_vehicle_speeds = current_speeds

    def get_metrics(self) -> dict[str, float]:
        current_running = self.eng.get_vehicle_count()

        throughput = len(self.all_seen_vehicles) - current_running

        current_time = self.eng.get_current_time()

        tracker_completed_att = calculate_mean(list(self.completed_travel_times.values()))

        truncated_travel_times = list(self.completed_travel_times.values())

        for v_id in self.last_active_vehicles:
            enter_time = self.vehicle_enter_time.get(v_id)

            if enter_time is not None:
                truncated_travel_times.append(current_time - enter_time)

        tracker_truncated_att = calculate_mean(truncated_travel_times)

        res = {
            "total_waiting_time_sum": self.waiting_time,
            "avg_waiting_time": self.waiting_time / len(self.vehicles) if len(self.vehicles) > 0 else 0.0,
            "avg_queue_length": (self.waiting_time / current_time if current_time > 0 else 0.0),
            "cityflow_completed_att": self.eng.get_average_travel_time(),
            "tracker_completed_att": tracker_completed_att,
            "tracker_truncated_att": tracker_truncated_att,
            "total_emergency_braking_count": float(self.emergency_braking_count),
            "emergency_braking_rate": (self.emergency_braking_count / len(self.vehicles) if len(self.vehicles) > 0 else 0.0),
            "throughput": float(throughput),
            "avg_queue_std": ((self.total_queue_std / self.step_count) if self.step_count > 0 else 0.0),
            "avg_speed": ((self.total_avg_speed / self.step_count) if self.step_count > 0 else 0.0),
        }

        if self.local_inter_id is not None:
            local_seen_v_nums = len(self.local_seen_vehicles)

            res.update(
                {
                    "local_seen_vehicle_count": local_seen_v_nums,
                    "local_wt": self.local_incoming_wt + self.local_outgoing_wt,
                    "local_awt": self.local_waiting_time / local_seen_v_nums if local_seen_v_nums > 0 else 0.0,
                    "local_incoming_awt": self.local_incoming_wt / local_seen_v_nums if local_seen_v_nums > 0 else 0.0,
                    "local_outgoing_awt": self.local_outgoing_wt / local_seen_v_nums if local_seen_v_nums > 0 else 0.0,
                    "local_ebc": float(self.local_ebc),
                    "local_ebr": self.local_ebc / local_seen_v_nums if local_seen_v_nums > 0 else 0.0,
                    "local_incoming_ebr": self.local_incoming_ebc / local_seen_v_nums if local_seen_v_nums > 0 else 0.0,
                    "local_outgoing_ebr": self.local_outgoing_ebc / local_seen_v_nums if local_seen_v_nums > 0 else 0.0,
                }
            )

        return res


class AsyncCityFlowEnv:
    def __init__(
        self,
        config: CityFlowConfig,
        yellow_phase: PhaseID = 0,
        yellow_duration: float = 5.0,
        phase_list_type: Literal["four", "eight"] = "four",
        max_simulation_seconds: Seconds = 3600.0,
        always_insert_yellow: bool = False,
        duration_includes_yellow: bool = False,
        obs_fn: ObsFunc | None = None,
        reward_fn: RewardFunc | None = None,
        local_inter_id: str | None = None,
    ) -> None:
        self.config: CityFlowConfig = config

        self.eng: CityFlowEngine = CityFlowEngine(config)

        self.log_dir = os.path.join(self.config.dir, "logs")

        os.makedirs(self.log_dir, exist_ok=True)

        self.yellow_phase = yellow_phase

        self.yellow_duration = yellow_duration

        self.always_insert_yellow = always_insert_yellow

        self.duration_includes_yellow = duration_includes_yellow

        if phase_list_type not in ("four", "eight"):
            raise ValueError(f"Invalid phase_list_type: {phase_list_type!r}")

        if phase_list_type == "four":
            self.phase_list: list[int] = [1, 2, 3, 4]

        elif phase_list_type == "eight":
            self.phase_list: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]

        self.phase_id_names = {
            0: "Yellow",
            1: "ETWT",
            2: "NTST",
            3: "ELWL",
            4: "NLSL",
            5: "ELET",
            6: "WLWT",
            7: "NLNT",
            8: "SLST",
        }

        self.max_simulation_seconds = max_simulation_seconds

        with open(Path(config.dir) / config.roadnetFile, encoding="utf-8") as f:
            self.roadnet: dict[str, Any] = json.load(f)

        self.inters_state = [
            IntersectionState(
                intersection_id=inter["id"],
                current_phase=0,
                next_action_time=0.0,
                last_action_time=0.0,
                is_yellow=False,
                yellow_end_time=0.0,
                min_green=10.0,
            )
            for inter in self.roadnet.get("intersections", [])
            if inter["virtual"] is False
        ]

        self.direction_short = {"North": "N", "South": "S", "East": "E", "West": "W"}

        self.direction_short_reverse = {"N": "North", "S": "South", "E": "East", "W": "West"}

        self.lane_direction_short = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

        self.location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

        self.location_incoming_dict = ["N", "S", "E", "W"]

        self.intersections_phases_lanes = self._init_phase_lanes()

        self.lane_properties = self._init_lane_properties()

        self.current_time = self.eng.get_current_time()

        self.obs_fn: ObsFunc = obs_fn if obs_fn is not None else self._default_get_obs

        self.reward_fn: RewardFunc = reward_fn if reward_fn is not None else self._default_get_reward

        self.waiting_vehicles = {}

        self.lane_vehicle_waiting_time = {}

        self._incoming_LT_lanes = [lane for lane in self.lane_properties if self.lane_properties[lane]["turn_direction"] in ["L", "T"]]

        if local_inter_id is not None:
            incoming_lanes = [lane for mv in self.intersections_phases_lanes[local_inter_id] for lane in self.intersections_phases_lanes[local_inter_id][mv]["from"]]

            outgoing_lanes = list(set([lane for mv in self.intersections_phases_lanes[local_inter_id] for lane in self.intersections_phases_lanes[local_inter_id][mv]["to"]]))

            self.metric_tracker = MetricsTracker(
                self.roadnet,
                self.eng,
                inter_id=local_inter_id,
                incoming_lanes=incoming_lanes,
                outgoing_lanes=outgoing_lanes,
            )

        else:
            self.metric_tracker = MetricsTracker(self.roadnet, self.eng)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.eng.reset(seed=False)

        if seed is not None:
            self.eng.set_random_seed(seed)

        self.current_time = self.eng.get_current_time()

        self.metric_tracker.reset()

        self.waiting_vehicles = {}

        self.lane_vehicle_waiting_time = {}

        self._init_intersection_tl_states()

        obs = {state.intersection_id: self.obs_fn(self.eng, state.intersection_id) for state in self.inters_state}

        return obs

    def step(
        self,
        actions: dict[str, AsyncAction],
    ) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        self._update_intersection_tl_states(actions)

        for state in self.inters_state:
            if state.is_yellow:
                self.eng.set_tl_phase(state.intersection_id, self.yellow_phase)

            else:
                self.eng.set_tl_phase(state.intersection_id, state.current_phase)

        target_time = self._compute_next_event_time()

        if target_time is None:
            return {}, {}, False, {"reason": "no_pending_events"}

        this_end_time = self.max_simulation_seconds if target_time is None else min(target_time, self.max_simulation_seconds)

        while self.current_time < this_end_time:
            self.eng.next_step()

            self.metric_tracker.update()

            self._update_lane_waiting_vehicles()

            self.current_time = self.eng.get_current_time()

            for state in self.inters_state:
                if state.is_yellow and abs(self.current_time - state.yellow_end_time) < 1e-9:
                    state.is_yellow = False

                    self.eng.set_tl_phase(state.intersection_id, state.current_phase)

                    self._log_phase_change(state.intersection_id, self.current_time, state.current_phase)

        ready_inters: list[IntersectionID] = self.find_ready_intersections()

        obs: dict[IntersectionID, Any] = {}

        rewards: dict[IntersectionID, float] = {}

        for inter_id in ready_inters:
            obs[inter_id] = self.obs_fn(self.eng, inter_id)

            rewards[inter_id] = self.reward_fn(self.eng, inter_id)

        done = abs(self.current_time - self.max_simulation_seconds) < 1e-9

        info: dict[str, Any] = {
            "current_time": self.current_time,
            "ready_intersections": ready_inters,
        }

        return obs, rewards, done, info

    def get_intersection_tl_state(self, intersection_id: IntersectionID) -> IntersectionState:
        for inter in self.inters_state:
            if inter.intersection_id == intersection_id:
                return inter

        raise ValueError(f"Intersection {intersection_id} not found.")

    def _init_intersection_tl_states(self) -> None:
        init_phase: int = 0

        for state in self.inters_state:
            state.current_phase = 1

            state.next_action_time = 0.0

            state.last_action_time = 0.0

            state.is_yellow = False

            state.yellow_end_time = 0.0

            state.min_green = 10.0

            self.eng.set_tl_phase(state.intersection_id, init_phase)

            self._reset_log_file(state.intersection_id)

            self._log_phase_change(state.intersection_id, 0.0, init_phase)

    def _update_intersection_tl_states(self, actions: dict[IntersectionID, AsyncAction]) -> None:
        for inter_id, action in actions.items():
            state = self.get_intersection_tl_state(inter_id)

            duration = action.duration

            if self.current_time == 0.0:
                state.next_action_time = self.current_time + duration

                state.current_phase = action.phase_id

                state.last_action_time = self.current_time

                state.is_yellow = False

            elif not self.always_insert_yellow and action.phase_id == state.current_phase:
                state.is_yellow = False

                state.yellow_end_time = 0.0

                state.next_action_time = max(state.next_action_time, self.current_time + duration)

            else:
                if self.duration_includes_yellow and duration <= self.yellow_duration:
                    raise ValueError("AsyncAction.duration must be greater than yellow_duration when duration_includes_yellow=True and a yellow phase is inserted.")

                state.is_yellow = True

                state.yellow_end_time = self.current_time + self.yellow_duration

                if self.duration_includes_yellow:
                    state.next_action_time = self.current_time + duration

                else:
                    state.next_action_time = state.yellow_end_time + duration

                state.current_phase = action.phase_id

                self._log_phase_change(state.intersection_id, self.current_time, self.yellow_phase)

            state.last_action_time = self.current_time

    def update_waiting_vehicles(self):
        vehicle_ids = self.eng.get_vehicles(include_waiting=False)

        for v_id in vehicle_ids:
            v_info = self.eng.get_vehicle_info(v_id)

            v_speed = v_info["speed"]

            if v_speed < 0.1:
                if v_id not in self.waiting_vehicles:
                    self.waiting_vehicles[v_id] = {"time": None, "lane": None}

                    self.waiting_vehicles[v_id]["time"] = self.config.interval

                    self.waiting_vehicles[v_id]["lane"] = v_info["drivable"]

                else:
                    if self.waiting_vehicles[v_id]["lane"] == v_info["drivable"]:
                        self.waiting_vehicles[v_id]["time"] += self.config.interval

                    else:
                        self.waiting_vehicles[v_id]["time"] = self.config.interval

                        self.waiting_vehicles[v_id]["lane"] = v_info["drivable"]

            else:
                if v_id in self.waiting_vehicles:
                    self.waiting_vehicles.pop(v_id)

            if v_id in self.waiting_vehicles and self.waiting_vehicles[v_id]["lane"] != v_info["drivable"]:
                self.waiting_vehicles.pop(v_id)

    def _update_lane_waiting_vehicles(self):
        lane_veh_list = self.eng.get_lane_vehicles()

        for lane in self._incoming_LT_lanes:
            lane_veh = lane_veh_list[lane]

            if lane not in self.lane_vehicle_waiting_time:
                self.lane_vehicle_waiting_time[lane] = {}

            for existing_v_id in list(self.lane_vehicle_waiting_time[lane].keys()):
                if existing_v_id not in lane_veh:
                    self.lane_vehicle_waiting_time[lane].pop(existing_v_id)

            if not lane_veh:
                continue

            for v_id in lane_veh:
                v_info = self.eng.get_vehicle_info(v_id)

                if v_info["speed"] < 0.1:
                    if v_id not in self.lane_vehicle_waiting_time[lane]:
                        self.lane_vehicle_waiting_time[lane][v_id] = 0.0

                    self.lane_vehicle_waiting_time[lane][v_id] += self.config.interval

    def get_final_metrics(self) -> dict[str, float]:
        return self.metric_tracker.get_metrics()

    def _reset_log_file(self, intersection_id: str):
        file_path = Path(self.log_dir) / f"{intersection_id}.txt"

        try:
            file_path.write_text("")

        except Exception as e:
            print(f"无法重置日志文件 {file_path}: {e}")

    def _log_phase_change(self, intersection_id: str, time: float, phase_id: int):
        file_path = Path(self.log_dir) / f"{intersection_id}.txt"

        with open(file_path, "a") as f:
            f.write(f"{time},{phase_id}\n")

    def _compute_next_event_time(self) -> Seconds | None:
        next_times: list[Seconds] = []

        for state in self.inters_state:
            if state.next_action_time > self.current_time:
                next_times.append(state.next_action_time)

        if not next_times:
            return None

        candidate = min(next_times)

        if abs(candidate - self.current_time) < 1e-9:
            return None

        return candidate

    def find_ready_intersections(self) -> list[IntersectionID]:
        ready: list[IntersectionID] = []

        for state in self.inters_state:
            if state.next_action_time is None or abs(state.next_action_time - self.current_time) < 1e-9:
                ready.append(state.intersection_id)

        return ready

    def _default_get_obs(self, eng: CityFlowEngine, intersection_id: IntersectionID) -> dict[str, Any]:
        lane_waiting: dict[str, int] = eng.get_lane_waiting_vehicle_count()

        lane_veh_cnt: dict[str, int] = eng.get_lane_vehicle_count()

        obs = {
            "time": float(eng.get_current_time()),
            "intersection_id": intersection_id,
            "total_waiting_vehicles": sum(lane_waiting.values()),
            "total_running_vehicles": sum(lane_veh_cnt.values()),
        }

        return obs

    def _default_get_reward(self, eng: CityFlowEngine, intersection_id: IntersectionID) -> float:
        lane_waiting: dict[str, int] = eng.get_lane_waiting_vehicle_count()

        total_waiting = sum(lane_waiting.values())

        return -float(total_waiting)

    def get_current_time(self) -> float:
        return self.eng.get_current_time()

    def get_intersection_ids(self) -> list[IntersectionID]:
        return [state.intersection_id for state in self.inters_state]

    def _init_phase_lanes(self) -> dict[str, dict]:
        direction_index_short = {0: "E", 1: "N", 2: "W", 3: "S"}

        tl_inter_phase_lanes = {}

        for inter in self.roadnet["intersections"]:
            if inter["virtual"]:
                continue

            inter_id = inter["id"]

            tl_inter_phase_lanes[inter_id] = {}

            tl = inter["trafficLight"]

            light_phases = tl["lightphases"]

            yellow_phase = light_phases[0]

            for phase in light_phases[1:]:
                avai_road_links = list(set(phase["availableRoadLinks"]) - set(yellow_phase["availableRoadLinks"]))

                phase_key = []

                for link_idx in avai_road_links:
                    road_link = inter["roadLinks"][tl["roadLinkIndices"][link_idx]]

                    road_link["startRoad"]

                    d_phase_key = direction_index_short[road_link["direction"]] + self.lane_direction_short[road_link["type"]]

                    tl_inter_phase_lanes[inter_id][d_phase_key] = {
                        "from": list(set([road_link["startRoad"] + "_" + str(lane_link["startLaneIndex"]) for lane_link in road_link["laneLinks"]])),
                        "to": list(set([road_link["endRoad"] + "_" + str(lane_link["endLaneIndex"]) for lane_link in road_link["laneLinks"]])),
                    }

                    phase_key.append(d_phase_key)

        return tl_inter_phase_lanes

    def _get_lane_turn_direction(self, lane_id: str) -> str:
        turn_directions = []

        for _inter, phases in self.intersections_phases_lanes.items():
            for phase_name, lanes in phases.items():
                if lane_id in lanes["from"]:
                    turn_directions.append(phase_name[1])

                    return turn_directions[0]

        return "R"

    def _init_lane_properties(self) -> dict[str, dict]:
        lane_properties = {}

        for road in self.roadnet["roads"]:
            road_id = road["id"]

            start_end_inter = [road["startIntersection"], road["endIntersection"]]

            road_length = calculate_road_length(road["points"])

            for index, lane in enumerate(road["lanes"]):
                full_lane_id = f"{road_id}_{index}"

                lane_properties[full_lane_id] = {
                    "max_speed": lane["maxSpeed"],
                    "length": road_length,
                    "turn_direction": self._get_lane_turn_direction(full_lane_id),
                    "start_end_inter": start_end_inter,
                }

        return lane_properties
