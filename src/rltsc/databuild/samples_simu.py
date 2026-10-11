import argparse
import concurrent.futures
import gc
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rltsc.trainer.parser import PHASE_ORDER

PHASE_TO_ID = {phase: index + 1 for index, phase in enumerate(PHASE_ORDER)}
RANKING_METRICS = ("global_queue_length", "local_queue_length", "local_awt", "att")
AVAILABLE_METRICS = (*RANKING_METRICS, "global_awt")


@dataclass(frozen=True)
class Metric:
    name: str
    direction: str = "min"
    weight: float = 1.0


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message.rstrip() + "\n")


def index_by_sample_id(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def parse_metric(value: str) -> Metric:
    parts = value.split(":")
    if len(parts) not in (2, 3) or parts[0] not in AVAILABLE_METRICS or parts[1] not in ("min", "max"):
        raise argparse.ArgumentTypeError(f"Invalid metric: {value}; expected NAME:min|max[:WEIGHT]")
    try:
        weight = float(parts[2]) if len(parts) == 3 else 1.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid weight: {value}") from exc
    if not math.isfinite(weight) or weight < 0:
        raise argparse.ArgumentTypeError("Metric weights must be finite and nonnegative")
    return Metric(parts[0], parts[1], weight)


def evaluate_phases(phase_results: list[dict[str, Any]], metrics: list[Metric], evaluation: str = "lexicographic", normalize: str = "none") -> dict[str, Any]:
    by_phase = {result["action_phase"]: result for result in phase_results}
    if len(phase_results) != len(PHASE_ORDER) or set(by_phase) != set(PHASE_ORDER):
        raise ValueError("Expected exactly one result for each of the four phases")
    if not metrics or len({metric.name for metric in metrics}) != len(metrics):
        raise ValueError("Metrics must be nonempty and unique")
    if evaluation not in ("lexicographic", "weighted") or normalize not in ("none", "minmax"):
        raise ValueError("Invalid evaluation or normalization")
    values = {}
    for metric in metrics:
        if metric.name not in AVAILABLE_METRICS or metric.direction not in ("min", "max") or not math.isfinite(metric.weight) or metric.weight < 0:
            raise ValueError(f"Invalid metric: {metric}")
        values[metric.name] = {phase: float(by_phase[phase][metric.name]) for phase in PHASE_ORDER}
        if not all(math.isfinite(value) for value in values[metric.name].values()):
            raise ValueError(f"Nonfinite metric: {metric.name}")
    if evaluation == "lexicographic":
        if normalize != "none" or any(metric.weight != 1.0 for metric in metrics):
            raise ValueError("Normalization and weights are only used in weighted evaluation")
        ranking = sorted(PHASE_ORDER, key=lambda phase: tuple(values[metric.name][phase] * (1 if metric.direction == "min" else -1) for metric in metrics))
        primary = metrics[0]
        returns = {phase: values[primary.name][phase] * (-1 if primary.direction == "min" else 1) for phase in PHASE_ORDER}
    else:
        if sum(metric.weight for metric in metrics) <= 0:
            raise ValueError("At least one metric must have positive weight")
        returns = {phase: 0.0 for phase in PHASE_ORDER}
        for metric in metrics:
            lower = min(values[metric.name].values())
            span = max(values[metric.name].values()) - lower
            for phase in PHASE_ORDER:
                value = values[metric.name][phase]
                if normalize == "minmax":
                    value = (value - lower) / span if span else 0.0
                returns[phase] += metric.weight * (-1 if metric.direction == "min" else 1) * value
        ranking = sorted(PHASE_ORDER, key=lambda phase: -returns[phase])
    if not all(math.isfinite(value) for value in returns.values()):
        raise ValueError("Nonfinite phase return")
    return {
        "best_phase": ranking[0],
        "phase_results": [by_phase[phase] for phase in PHASE_ORDER],
        "phase_ranking": ranking,
        "ranking_metrics": [metric.name for metric in metrics],
        "phase_return": returns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--batch-root")
    source.add_argument("--input")
    parser.add_argument("--roadnet", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation", choices=("lexicographic", "weighted"), default="lexicographic")
    parser.add_argument("--metric", type=parse_metric, action="append")
    parser.add_argument("--normalize", choices=("none", "minmax"), default="none")
    parser.add_argument("--duration", type=float, default=75.0)
    parser.add_argument("--phase-duration", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow-phase-id", type=int, default=0)
    parser.add_argument("--yellow-duration", type=float, default=5.0)
    parser.add_argument("--always-insert-yellow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--duration-includes-yellow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lane-change", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-vehicles", type=int, default=1)
    parser.add_argument("--phase-difference-min", type=float, default=2.0)
    parser.add_argument("--phase-difference-ratio", type=float, default=2.0)
    parser.add_argument("--high-demand-min", type=float, default=10.0)
    parser.add_argument("--high-demand-ratio", type=float, default=1.5)
    parser.add_argument("--demand-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.evaluation == "weighted" and not args.metric:
        parser.error("Weighted evaluation requires at least one --metric")
    args.metric = args.metric if args.metric else [Metric(name) for name in RANKING_METRICS]
    if args.evaluation == "lexicographic" and (args.normalize != "none" or any(metric.weight != 1.0 for metric in args.metric)):
        parser.error("Normalization and weights require --evaluation weighted")
    if args.evaluation == "weighted" and sum(metric.weight for metric in args.metric) <= 0:
        parser.error("At least one metric must have positive weight")
    if len({metric.name for metric in args.metric}) != len(args.metric):
        parser.error("Duplicate metrics")
    for name in ("duration", "phase_duration", "interval", "phase_difference_min", "phase_difference_ratio", "high_demand_min", "high_demand_ratio"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.yellow_duration) or args.yellow_duration < 0 or args.workers < 1 or args.min_vehicles < 0 or (args.limit is not None and args.limit < 0):
        parser.error("Invalid duration, workers, vehicle threshold, or limit")
    return args


def collect_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.input:
        rows = read_json(args.input)
        if not isinstance(rows, list):
            raise ValueError("Input samples must be a JSON array")
        return (rows[: args.limit] if args.limit is not None else rows), []
    if args.limit == 0:
        return [], []
    if args.run_dir:
        runs = [Path(args.run_dir).resolve()]
    else:
        output_dir = Path(args.output_dir).resolve()
        runs = sorted(path for path in Path(args.batch_root).resolve().iterdir() if path.is_dir() and path.resolve() != output_dir and path.name != "data_build")
    kept, rejected = [], []
    for run in runs:
        archives = run / "archives"
        logs = run / "action_logs"
        run_name = str(read_json(run / "simulation_result.json")["traj_name"])
        if not archives.is_dir() or not logs.is_dir():
            raise FileNotFoundError(f"Missing action logs or archives: {run}")
        for log_file in sorted(logs.glob("*_action_log.json")):
            entries = read_json(log_file)
            if not isinstance(entries, list):
                raise ValueError(f"Action log must be an array: {log_file}")
            intersection_id = log_file.name.removesuffix("_action_log.json")
            for index, entry in enumerate(entries):
                try:
                    timestamp = int(float(entry["timestamp"]))
                    snapshot = archives / str(timestamp) / "snapshot.json"
                    states = archives / str(timestamp) / "inter_states.json"
                    read_json(snapshot)
                    if not states.is_file():
                        raise FileNotFoundError(f"Missing intersection states: {states}")
                    messages = []
                    for message in entry["llm_call_log"]["messages"]:
                        if message["role"] == "assistant":
                            break
                        messages.append(message)
                    if len(messages) < 2:
                        raise ValueError("Missing prompt messages")
                    phase_state = entry["phase_state"]
                    total = sum(int(phase_state[phase]["vehicle_count"]) for phase in PHASE_ORDER)
                    if total < args.min_vehicles:
                        raise ValueError("low_information_state")
                    if args.demand_filter:
                        pressures = sorted(float(phase_state[phase]["pressure"]) for phase in PHASE_ORDER)
                        queues = sorted(float(phase_state[phase]["queue"]) for phase in PHASE_ORDER)
                        if queues[-2] == 0 or pressures[-2] == 0:
                            raise ValueError("zero_phase_state")
                        if queues[-2] >= args.phase_difference_min and pressures[-2] >= args.phase_difference_min:
                            if pressures[-1] / pressures[-2] > args.phase_difference_ratio and queues[-1] / queues[-2] > args.phase_difference_ratio:
                                raise ValueError("high_phase_pressure_diff")
                        if queues[-2] >= args.high_demand_min and pressures[-2] >= args.high_demand_min:
                            if pressures[-1] / pressures[-2] >= args.high_demand_ratio and queues[-1] / queues[-2] >= args.high_demand_ratio:
                                raise ValueError("higher_phase_pressure_diff")
                    kept.append(
                        {
                            "sample_id": f"{run_name}::{intersection_id}::{timestamp}",
                            "intersection_id": intersection_id,
                            "timestamp": timestamp,
                            "prompt": messages,
                            "prompt_state_dict": entry["prompt_state_dict"],
                            "state": phase_state,
                            "current_phase": entry["phase"],
                            "snapshot_path": str(snapshot),
                            "inter_states_path": str(states),
                        }
                    )
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    rejected.append({"run_dir": str(run), "log_file": str(log_file), "entry_index": index, "reject_reason": str(exc)})
                if args.limit is not None and len(kept) >= args.limit:
                    return kept, rejected
    return kept, rejected


def queue_policy_action(env: Any, intersection_id: str, duration: float) -> Any:
    from rltsc.atlm.async_cityflow_env import AsyncAction

    waiting = env.eng.get_lane_waiting_vehicle_count()
    scores = {
        phase: sum(int(waiting[lane]) for movement in (phase[:2], phase[-2:]) for lane in env.intersections_phases_lanes[intersection_id][movement]["from"])
        for phase in PHASE_ORDER
    }
    return AsyncAction(phase_id=PHASE_TO_ID[max(PHASE_ORDER, key=lambda phase: scores[phase])], duration=duration)


def local_queue_length(env: Any, intersection_id: str) -> int:
    waiting = env.eng.get_lane_waiting_vehicle_count()
    lanes = {lane for movement in env.intersections_phases_lanes[intersection_id].values() for side in ("from", "to") for lane in movement[side]}
    return sum(int(waiting[lane]) for lane in lanes)


def run_one_phase(sample: dict[str, Any], phase: str, options: dict[str, Any]) -> dict[str, Any]:
    from rltsc.atlm.async_cityflow_env import AsyncAction, AsyncCityFlowEnv, IntersectionState
    from rltsc.atlm.configs import CityFlowConfig

    with tempfile.TemporaryDirectory(prefix="databuild_simu_") as temporary:
        shutil.copyfile(options["roadnet"], Path(temporary) / "roadnet.json")
        shutil.copyfile(options["flow"], Path(temporary) / "flow.json")
        config = CityFlowConfig(
            dir=temporary, roadnetFile="roadnet.json", flowFile="flow.json", interval=options["interval"], seed=options["seed"], laneChange=options["lane_change"]
        )
        env = AsyncCityFlowEnv(
            config=config,
            yellow_phase=options["yellow_phase_id"],
            yellow_duration=options["yellow_duration"],
            always_insert_yellow=options["always_insert_yellow"],
            duration_includes_yellow=options["duration_includes_yellow"],
            phase_list_type="four",
            local_inter_id=sample["intersection_id"],
        )
        try:
            env.reset()
            env.eng.load_from_file(str(Path(sample["snapshot_path"]).resolve()))
            env.inters_state = [IntersectionState(**state) for state in read_json(sample["inter_states_path"])]
            env.current_time = env.eng.get_current_time()
            env.max_simulation_seconds = env.current_time + options["duration"]
            applied, done = False, False
            queue_total, local_queue_total, steps = 0.0, 0.0, 0
            while not done and env.current_time < env.max_simulation_seconds:
                ready = env.find_ready_intersections()
                if not ready:
                    break
                actions = {}
                for intersection in ready:
                    if intersection == sample["intersection_id"] and not applied:
                        actions[intersection] = AsyncAction(phase_id=PHASE_TO_ID[phase], duration=options["phase_duration"])
                        applied = True
                    else:
                        actions[intersection] = queue_policy_action(env, intersection, options["phase_duration"])
                previous_time = env.current_time
                _, _, done, _ = env.step(actions)
                if env.current_time <= previous_time:
                    raise ValueError("Simulation did not advance")
                queue_total += sum(int(value) for value in env.eng.get_lane_waiting_vehicle_count().values())
                local_queue_total += local_queue_length(env, sample["intersection_id"])
                steps += 1
            if not applied or steps == 0:
                raise ValueError("Target phase was not evaluated")
            result = env.get_final_metrics()
            return {
                "action_phase": phase,
                "local_awt": float(result["local_awt"]),
                "global_awt": float(result["avg_waiting_time"]),
                "local_queue_length": local_queue_total / steps,
                "global_queue_length": queue_total / steps,
                "att": float(env.eng.get_average_travel_time()),
                "target_applied": applied,
            }
        finally:
            del env
            gc.collect()


def simulate_sample(sample: dict[str, Any], options: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        phase_results = [run_one_phase(sample, phase, options) for phase in PHASE_ORDER]
        simulation = evaluate_phases(phase_results, options["metric"], options["evaluation"], options["normalize"])
        prompt = sample["prompt"] if "prompt" in sample else sample["messages"]
        result = dict(sample)
        result.pop("messages", None)
        result["prompt"] = prompt
        result["simulation"] = simulation
        return result, None
    except Exception as exc:
        return None, {"sample_id": sample["sample_id"], "reject_reason": str(exc)}


def main() -> None:
    args = parse_args()
    options = vars(args).copy()
    options["roadnet"] = str(Path(args.roadnet).resolve())
    options["flow"] = str(Path(args.flow).resolve())
    for name in ("roadnet", "flow"):
        if not Path(options[name]).is_file():
            raise FileNotFoundError(options[name])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "simu_log.txt"
    log.write_text("", encoding="utf-8")
    samples, filtered = collect_samples(args)
    index_by_sample_id(samples, "simulation input")
    write_json(output / "filter_rejected.json", filtered)
    kept, rejected = [], []
    write_json(output / "origin_simu.json", kept)
    write_json(output / "simu_rejected.json", rejected)
    started = time.monotonic()

    def save(result: tuple[dict[str, Any] | None, dict[str, Any] | None]) -> None:
        row, failure = result
        if row is not None:
            kept.append(row)
        if failure is not None:
            rejected.append(failure)
        write_json(output / "origin_simu.json", kept)
        write_json(output / "simu_rejected.json", rejected)
        message = f"completed={len(kept) + len(rejected)}/{len(samples)} kept={len(kept)} rejected={len(rejected)} elapsed={time.monotonic() - started:.1f}s"
        append_log(log, message)
        print(message, flush=True)

    if args.workers == 1:
        for sample in samples:
            save(simulate_sample(sample, options))
    elif samples:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            tasks = [executor.submit(simulate_sample, sample, options) for sample in samples]
            for task in concurrent.futures.as_completed(tasks):
                save(task.result())
    print(f"kept={len(kept)} rejected={len(rejected)} filtered={len(filtered)} output={output}")


if __name__ == "__main__":
    main()
