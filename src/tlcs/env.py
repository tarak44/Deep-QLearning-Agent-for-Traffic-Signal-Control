from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from tlcs.constants import (
    ACTION_TO_DURATION_IDX,
    ACTION_TO_TL_PHASE,
    BASE_STATE_SIZE,
    CELLS_PER_LANE_GROUP,
    INCOMING_EDGES,
    LANE_DISTANCE_TO_CELL,
    LANE_ID_TO_GROUP,
    NUM_LANE_GROUPS,
    ROAD_MAX_LENGTH,
    STATE_SIZE,
    TL_GREEN_TO_YELLOW,
    TRAFFIC_LIGHT_ID,
)
from tlcs.generator import generate_routefile

STOP_SPEED_THRESHOLD = 0.1


@dataclass
class EnvStats:
    """Snapshot of environment statistics for a single simulation step."""

    queue_length: int
    max_queue: int


class Environment:
    """Reinforcement-learning environment wrapper around a SUMO traffic simulation."""

    def __init__(  # noqa: PLR0913
        self,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        demand_profile: str,
        peak_start: float,
        peak_end: float,
        peak_share: float,
        green_duration_multipliers: list[float],
        incident_prob: float,
        incident_duration: int,
        incident_speed_factor: float,
        n_pedestrians: int,
        sumocfg_file: Path,
        gui: bool,
    ) -> None:
        """Initialize the environment.

        Args:
            n_cars_generated: Number of cars to generate for the episode.
            max_steps: Maximum number of simulation steps in an episode.
            yellow_duration: Number of steps to hold a yellow phase.
            green_duration: Number of steps to hold a green phase.
            turn_chance: Probability for each car to turn instead of going straight.
            demand_profile: Traffic demand profile to use.
            peak_start: Peak window start (normalized 0-1).
            peak_end: Peak window end (normalized 0-1).
            peak_share: Fraction of vehicles generated during the peak window.
            green_duration_multipliers: Multipliers for short/long green duration.
            incident_prob: Probability of a random slowdown per step.
            incident_duration: Duration of an incident in steps.
            incident_speed_factor: Speed multiplier during incident.
            n_pedestrians: Number of pedestrians per episode.
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to use the SUMO GUI binary.
        """
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.demand_profile = demand_profile
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.peak_share = peak_share
        self.green_duration_multipliers = green_duration_multipliers
        self.incident_prob = incident_prob
        self.incident_duration = incident_duration
        self.incident_speed_factor = incident_speed_factor
        self.n_pedestrians = n_pedestrians
        self.sumocfg_file = sumocfg_file
        self.gui = gui

        self.step = 0
        self._incident_remaining: dict[str, int] = {}
        self._rng = np.random.default_rng()

    def build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line based on configuration settings.

        Returns:
            List of command-line arguments to start SUMO.
        """
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")

        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        return [
            sumo_binary,
            "-c",
            str(self.sumocfg_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
            "--ignore-route-errors",
            "true" if self.n_pedestrians > 0 else "false",
        ]

    def activate(self) -> None:
        """Start the SUMO simulation."""
        sumo_cmd = self.build_sumo_cmd()
        traci.start(sumo_cmd)

    def deactivate(self) -> None:
        """Stop the SUMO simulation."""
        traci.close()

    def is_over(self) -> bool:
        """Check whether the maximum number of steps has been reached.

        Returns:
            True if the episode is finished, False otherwise.
        """
        return self.step >= self.max_steps

    def generate_routefile(self, seed: int) -> None:
        """Generate a route file for the current episode.

        Args:
            seed: Random seed used for route generation.
        """
        generate_routefile(
            seed=seed,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
            turn_chance=self.turn_chance,
            demand_profile=self.demand_profile,
            peak_start=self.peak_start,
            peak_end=self.peak_end,
            peak_share=self.peak_share,
            n_pedestrians=self.n_pedestrians,
        )

    def _get_lane_cell(self, lane_pos: float) -> int:
        """Map a continuous lane position to a discrete cell index.

        The lane is inverted so that 0 is at the traffic light and clamped to [0, ROAD_MAX_LENGTH].

        Args:
            lane_pos: Distance from the start of the edge in meters.

        Returns:
            Index of the discretized cell (0-based).
        """
        # invert so 0 is at the light; clamp to [0, ROAD_MAX_LENGTH]
        lane_pos = ROAD_MAX_LENGTH - lane_pos
        lane_pos = max(0.0, min(ROAD_MAX_LENGTH, lane_pos))

        for distance, cell in LANE_DISTANCE_TO_CELL.items():
            if lane_pos <= distance:
                return cell

        msg = "Error while getting lane cell."
        raise RuntimeError(msg)

    def get_state(self) -> NDArray:
        """Compute the discrete state representation of all vehicles.

        The state includes a binary occupancy grid (lane groups x cells) plus per-lane-group
        queue length and average speed features.

        Returns:
            A NumPy array of shape (state_size,) with 0/1 occupancy values.
        """
        state = np.zeros(STATE_SIZE, dtype=float)
        queue_counts = np.zeros(NUM_LANE_GROUPS, dtype=float)
        speed_sums = np.zeros(NUM_LANE_GROUPS, dtype=float)
        speed_counts = np.zeros(NUM_LANE_GROUPS, dtype=float)

        for car_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_group = LANE_ID_TO_GROUP.get(lane_id)
            if lane_group is None:
                # Ignore cars that are not on incoming lanes.
                continue

            lane_pos: float = traci.vehicle.getLanePosition(car_id)
            lane_cell = self._get_lane_cell(lane_pos)

            car_position = lane_group * CELLS_PER_LANE_GROUP + lane_cell

            if car_position < 0 or car_position >= BASE_STATE_SIZE:
                msg = "Out of bounds car position."
                raise ValueError(msg)

            state[car_position] = 1.0
            speed = float(traci.vehicle.getSpeed(car_id))
            max_speed = float(traci.lane.getMaxSpeed(lane_id))
            if max_speed > 0:
                speed_sums[lane_group] += speed / max_speed
                speed_counts[lane_group] += 1
            if speed < STOP_SPEED_THRESHOLD:
                queue_counts[lane_group] += 1

        queue_offset = BASE_STATE_SIZE
        speed_offset = BASE_STATE_SIZE + NUM_LANE_GROUPS
        denom = max(1, self.n_cars_generated)
        for i in range(NUM_LANE_GROUPS):
            state[queue_offset + i] = queue_counts[i] / denom
            if speed_counts[i] > 0:
                state[speed_offset + i] = speed_sums[i] / speed_counts[i]

        return state

    def get_cumulated_waiting_time(self) -> float:
        """Compute the sum of waiting times for vehicles on incoming edges.

        Returns:
            Total accumulated waiting time of all vehicles on incoming edges.
        """
        waiting_times = 0.0

        for car_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id not in INCOMING_EDGES:
                continue
            wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
            waiting_times += wait_time

        return waiting_times

    def _set_yellow_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the yellow phase corresponding to a green phase.

        Args:
            green_phase_code: Code of the current green phase.
        """
        yellow_phase_code = TL_GREEN_TO_YELLOW[green_phase_code]
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_code)

    def _set_green_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the given green phase.

        Args:
            green_phase_code: Code of the green phase to activate.
        """
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, green_phase_code)

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance the simulation for a given number of steps.

        The actual number of steps is capped so as not to exceed `max_steps`.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            A list of EnvStats, one entry per simulation step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            self._apply_incidents()
            traci.simulationStep()
            self.step += 1
            queue_length, max_queue = self.get_queue_stats()
            stats.append(EnvStats(queue_length=queue_length, max_queue=max_queue))

        return stats

    def _apply_incidents(self) -> None:
        """Optionally slow random vehicles to simulate incidents."""
        if self.incident_prob <= 0:
            return

        # Decrease timers and restore speed when done.
        finished = []
        for car_id, remaining in list(self._incident_remaining.items()):
            new_remaining = remaining - 1
            if new_remaining <= 0:
                finished.append(car_id)
            else:
                self._incident_remaining[car_id] = new_remaining

        for car_id in finished:
            with suppress(traci.TraCIException):
                traci.vehicle.setSpeed(car_id, -1)
            self._incident_remaining.pop(car_id, None)

        # Possibly introduce a new incident.
        if self._rng.random() < self.incident_prob:
            vehicles = traci.vehicle.getIDList()
            if not vehicles:
                return
            car_id = str(self._rng.choice(vehicles))
            if car_id in self._incident_remaining:
                return
            with suppress(traci.TraCIException):
                speed = float(traci.vehicle.getSpeed(car_id))
                traci.vehicle.setSpeed(car_id, speed * self.incident_speed_factor)
                self._incident_remaining[car_id] = self.incident_duration

    def execute(self, action: int) -> list[EnvStats]:
        """Execute an action by changing the traffic light phase.

        If the requested phase differs from the current one, a yellow phase is inserted before
        switching to the new green phase.

        Args:
            action: Discrete action index mapped to a traffic light phase.

        Returns:
            A list of EnvStats collected during the applied phases.
        """
        next_green_phase = ACTION_TO_TL_PHASE[action]
        duration_idx = ACTION_TO_DURATION_IDX[action]
        duration_multiplier = self.green_duration_multipliers[duration_idx]
        green_duration = max(1, round(self.green_duration * duration_multiplier))
        current_green_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

        stats: list[EnvStats] = []

        if next_green_phase != current_green_phase:
            self._set_yellow_phase(current_green_phase)
            stats_yellow = self._simulate(self.yellow_duration)
            stats.extend(stats_yellow)

        if self.is_over():
            return stats

        self._set_green_phase(next_green_phase)
        stats_green = self._simulate(green_duration)
        stats.extend(stats_green)

        return stats

    def get_queue_stats(self) -> tuple[int, int]:
        """Return total and max stopped vehicles on incoming edges.

        Returns:
            Tuple of (total queue length, max per-edge queue length).
        """
        halt_n = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_s = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_e = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_w = traci.edge.getLastStepHaltingNumber("W2TL")
        total = int(halt_n + halt_s + halt_e + halt_w)
        max_queue = max(halt_n, halt_s, halt_e, halt_w)
        return total, max_queue
