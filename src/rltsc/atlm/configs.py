from dataclasses import dataclass, field
from typing import Literal, TypedDict


@dataclass
class CityFlowConfig:
    interval: float = field(default=1.0, metadata={"help": "Time of each simulation step (in seconds)."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    dir: str = field(
        default=".",
        metadata={"help": "Root directory, all file path will be relative to this directory."},
    )

    roadnetFile: str = field(default="roadnet.json", metadata={"help": "Path for roadnet file."})

    flowFile: str = field(default="flow.json", metadata={"help": "Path for flow file."})

    rlTrafficLight: bool = field(
        default=True,
        metadata={"help": "Whether to enable traffic light control through python API. If set to false, default traffic light plan defined in roadnetFile will be used."},
    )

    saveReplay: bool = field(
        default=False,
        metadata={"help": "Whether to save simulation for replay. If set to true, roadnetLogFile and replayLogFile are required."},
    )

    roadnetLogFile: str = field(
        default="roadnetLog.json",
        metadata={"help": "Path for roadnet replay file. This is a special roadnet file for replay, not the same as roadnetFile."},
    )

    replayLogFile: str = field(
        default="replayLog.txt",
        metadata={"help": "Path for replay. This file contains vehicle positions and traffic light situation of the simulation."},
    )

    laneChange: bool = field(
        default=False,
        metadata={"help": "Whether to enable lane changing behavior for vehicles."},
    )


@dataclass
class VehicleInfo(TypedDict):
    running: bool

    speed: float

    distance: float

    drivable: str

    road: str

    intersection: str

    route: str


@dataclass
class SimulatorConfig:
    phase_list_type: Literal["four", "eight"] = field(
        default="four",
        metadata={"help": "The type of traffic light phase list, 'four' for 4-phase, 'eight' for 8-phase."},
    )

    yellow_duration: int = field(
        default=5,
        metadata={"help": "Duration of yellow light phase (in seconds)."},
    )

    yellow_phase_id: int = field(
        default=0,
        metadata={"help": "The phase ID for the yellow light phase."},
    )

    always_insert_yellow: bool = field(
        default=False,
        metadata={"help": "Whether to always insert yellow light phase between phase changes."},
    )

    duration_includes_yellow: bool = field(
        default=True,
        metadata={
            "help": (
                "Controls the meaning of fixed_phase_duration / AsyncAction.duration. "
                "False keeps the legacy behavior: duration is pure green time and a phase switch lasts "
                "yellow_duration + duration. True matches CoLLMLight: duration is the total control "
                "step and yellow time is included inside it. For CoLLMLight-compatible evaluation, use "
                "duration_includes_yellow=True with always_insert_yellow=False."
            )
        },
    )

    max_simulation_seconds: int = field(
        default=3600,
        metadata={"help": "Maximum duration of the simulation (in seconds)."},
    )

    strategy: str = field(
        default="fixs_adad",
        metadata={"help": "The strategy for traffic light control. Options are 'fo' (Fixed Order), 'dur' (Duration Based), and 'dur_optimized' (Optimized Duration Based)."},
    )

    fixed_phase_duration: float = field(
        default=20.0,
        metadata={"help": "Fixed duration for each traffic light phase (in seconds)."},
    )

    strategy_params: dict = field(
        default_factory=dict,
        metadata={"help": "Additional parameters for the strategy."},
    )

    save_snapshot: bool = field(
        default=False,
        metadata={"help": "Whether to save snapshots during the simulation."},
    )


@dataclass
class LLMConfig:
    llm_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "Base URL for the LLM API."},
    )

    llm_api_key: str | None = field(
        default=None,
        metadata={"help": "API key for the LLM."},
    )

    llm_model_name: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "Model name for the LLM."},
    )
