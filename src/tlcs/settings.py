from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator


class TrainingSettings(BaseModel):
    """Configuration options for training the RL agent."""

    # simulation
    gui: bool
    total_episodes: PositiveInt
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    green_duration: PositiveInt
    yellow_duration: PositiveInt
    turn_chance: Annotated[float, Field(ge=0, le=1)]
    demand_profile: str = "flat"
    peak_start: Annotated[float, Field(ge=0, le=1)] = 0.3
    peak_end: Annotated[float, Field(ge=0, le=1)] = 0.7
    peak_share: Annotated[float, Field(ge=0, le=1)] = 0.5
    n_pedestrians: NonNegativeInt = 0
    incident_prob: Annotated[float, Field(ge=0, le=1)] = 0.0
    incident_duration: PositiveInt = 5
    incident_speed_factor: Annotated[float, Field(gt=0, le=1)] = 0.3

    # model
    num_layers: PositiveInt
    width_layers: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    training_epochs: PositiveInt

    # memory
    memory_size_min: NonNegativeInt
    memory_size_max: PositiveInt

    # agent
    gamma: Annotated[float, Field(ge=0, le=1)]
    use_double_dqn: bool = True
    target_update_interval: PositiveInt = 50
    grad_clip_norm: PositiveFloat = 10.0
    seed: int | None = None
    checkpoint_interval: PositiveInt = 10
    queue_penalty_weight: PositiveFloat = 0.1
    max_queue_penalty_weight: PositiveFloat = 0.2
    green_duration_multipliers: list[PositiveFloat] = [0.5, 1.0]

    # paths
    sumocfg_file: Path

    @model_validator(mode="after")
    def check_memory_bounds(self) -> Self:
        """Ensure that memory_size_min is strictly smaller than memory_size_max."""
        if self.memory_size_min >= self.memory_size_max:
            msg = (
                f"memory_size_min ({self.memory_size_min}) must be smaller than "
                f"memory_size_max ({self.memory_size_max})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_peak_bounds(self) -> Self:
        """Ensure peak demand bounds are valid when enabled."""
        if self.demand_profile not in {"flat", "peak"}:
            msg = "demand_profile must be 'flat' or 'peak'"
            raise ValueError(msg)
        if self.peak_start >= self.peak_end:
            msg = "peak_start must be smaller than peak_end"
            raise ValueError(msg)
        if len(self.green_duration_multipliers) != 2:
            msg = "green_duration_multipliers must contain exactly 2 values"
            raise ValueError(msg)
        return self


class TestingSettings(BaseModel):
    """Configuration options for testing a trained RL agent."""

    # simulation
    gui: bool
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    episode_seed: int
    yellow_duration: PositiveInt
    green_duration: PositiveInt
    turn_chance: Annotated[float, Field(ge=0, le=1)]
    demand_profile: str = "flat"
    peak_start: Annotated[float, Field(ge=0, le=1)] = 0.3
    peak_end: Annotated[float, Field(ge=0, le=1)] = 0.7
    peak_share: Annotated[float, Field(ge=0, le=1)] = 0.5
    n_pedestrians: NonNegativeInt = 0

    # agent
    gamma: Annotated[float, Field(ge=0, le=1)]
    queue_penalty_weight: PositiveFloat = 0.1
    max_queue_penalty_weight: PositiveFloat = 0.2
    green_duration_multipliers: list[PositiveFloat] = [0.5, 1.0]
    incident_prob: Annotated[float, Field(ge=0, le=1)] = 0.0
    incident_duration: PositiveInt = 5
    incident_speed_factor: Annotated[float, Field(gt=0, le=1)] = 0.3

    # paths
    sumocfg_file: Path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        msg = f"Invalid YAML format in {path}; expected a mapping at the top level"
        raise TypeError(msg)

    return data


def load_training_settings(settings_file: Path) -> TrainingSettings:
    """Load and validate training settings from a YAML file.

    Args:
        settings_file: Path to the training settings YAML file.

    Returns:
        A validated TrainingSettings instance.
    """
    return TrainingSettings.model_validate(load_yaml(settings_file))


def load_testing_settings(settings_file: Path) -> TestingSettings:
    """Load and validate testing settings from a YAML file.

    Args:
        settings_file: Path to the testing settings YAML file.

    Returns:
        A validated TestingSettings instance.
    """
    return TestingSettings.model_validate(load_yaml(settings_file))
