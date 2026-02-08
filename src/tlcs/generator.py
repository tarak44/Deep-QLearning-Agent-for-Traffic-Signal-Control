import numpy as np
from numpy.typing import NDArray

from tlcs.constants import (
    ROUTES_FILE,
    ROUTES_FILE_HEADER,
    STRAIGHT_ROUTES,
    TURN_ROUTES,
    VEHICLE_TYPE_WEIGHTS,
    VEHICLE_TYPES,
)

OFF_PEAK_SPLIT = 0.5


def _map_to_interval(values: NDArray, new_min: int, new_max: int) -> NDArray:
    """Linearly map values to the interval [new_min, new_max].

    Falls back to a constant array if the input has zero range.

    Args:
        values: Input array of values to be re-scaled.
        new_min: Minimum value of the target interval.
        new_max: Maximum value of the target interval.

    Returns:
        Array of values mapped to the target interval, same shape as `values`.
    """
    old_min = float(values.min())
    old_max = float(values.max())
    return np.interp(values, (old_min, old_max), (new_min, new_max))


def _get_car_row(route_id: str, car_i: int, step: int, vtype: str) -> str:
    """Build the XML row describing a single vehicle.

    Args:
        route_id: Identifier of the route the vehicle will follow.
        car_i: Index of the vehicle in the episode.
        step: Simulation step at which the vehicle departs.
        vtype: Vehicle type identifier.

    Returns:
        XML snippet representing the vehicle element.
    """
    return f'    <vehicle id="{route_id}_{car_i}" type="{vtype}" route="{route_id}" depart="{step}" departLane="random" departSpeed="10" />'  # noqa: E501


def generate_routefile(  # noqa: PLR0913
    seed: int,
    n_cars_generated: int,
    max_steps: int,
    turn_chance: float,
    demand_profile: str = "flat",
    peak_start: float = 0.3,
    peak_end: float = 0.7,
    peak_share: float = 0.5,
    n_pedestrians: int = 0,
) -> None:
    """Generate a SUMO route file for one simulation episode.

    Car departure times follow a Weibull distribution re-scaled to [0, max_steps].
    A fraction of cars go straight and the rest turn, controlled by turn_chance.

    Args:
        seed: Random seed for reproducible generation.
        n_cars_generated: Number of cars to generate in the episode.
        max_steps: Maximum simulation step for car departures.
        turn_chance: Probability to select a turn route rather than a straight route.
        demand_profile: Traffic demand profile ("flat" or "peak").
        peak_start: Peak window start (normalized 0-1).
        peak_end: Peak window end (normalized 0-1).
        peak_share: Fraction of vehicles generated during the peak window.
        n_pedestrians: Number of pedestrians to generate.
    """
    rng = np.random.default_rng(seed)

    if demand_profile == "flat":
        timings = np.sort(rng.weibull(2.0, size=n_cars_generated))
        generated_steps = _map_to_interval(timings, new_min=0, new_max=max_steps)
        depart_steps = np.rint(generated_steps).astype(int)
    elif demand_profile == "peak":
        peak_count = int(n_cars_generated * peak_share)
        off_count = n_cars_generated - peak_count

        peak_start_step = int(peak_start * max_steps)
        peak_end_step = int(peak_end * max_steps)

        peak_timings = np.sort(rng.weibull(2.0, size=peak_count))
        peak_steps = _map_to_interval(peak_timings, new_min=peak_start_step, new_max=peak_end_step)

        off_timings = rng.weibull(2.0, size=off_count)
        off_steps = np.zeros(off_count, dtype=float)
        for i in range(off_count):
            if rng.random() < OFF_PEAK_SPLIT:
                off_steps[i] = _map_to_interval(np.array([off_timings[i]]), 0, peak_start_step)[0]
            else:
                off_steps[i] = _map_to_interval(
                    np.array([off_timings[i]]),
                    peak_end_step,
                    max_steps,
                )[0]

        depart_steps = np.rint(np.concatenate([peak_steps, off_steps])).astype(int)
        rng.shuffle(depart_steps)
    else:
        msg = f"Unknown demand_profile: {demand_profile}"
        raise ValueError(msg)

    ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with ROUTES_FILE.open("w", encoding="utf-8") as routes_file:
        print(ROUTES_FILE_HEADER, file=routes_file)

        if n_pedestrians > 0:
            ped_per_flow = max(1, n_pedestrians // 4)
            period = max(1, int(max_steps / ped_per_flow))

            print(
                f'    <personFlow id="ped_N" begin="0" end="{max_steps}" period="{period}">',
                file=routes_file,
            )
            print(
                '        <walk edges="N2TL"/>',
                file=routes_file,
            )
            print(
                "    </personFlow>",
                file=routes_file,
            )
            print(
                f'    <personFlow id="ped_S" begin="0" end="{max_steps}" period="{period}">',
                file=routes_file,
            )
            print(
                '        <walk edges="S2TL"/>',
                file=routes_file,
            )
            print(
                "    </personFlow>",
                file=routes_file,
            )
            print(
                f'    <personFlow id="ped_E" begin="0" end="{max_steps}" period="{period}">',
                file=routes_file,
            )
            print(
                '        <walk edges="E2TL"/>',
                file=routes_file,
            )
            print(
                "    </personFlow>",
                file=routes_file,
            )
            print(
                f'    <personFlow id="ped_W" begin="0" end="{max_steps}" period="{period}">',
                file=routes_file,
            )
            print(
                '        <walk edges="W2TL"/>',
                file=routes_file,
            )
            print(
                "    </personFlow>",
                file=routes_file,
            )

        for car_i, step in enumerate(depart_steps):
            routes_selected = TURN_ROUTES if rng.random() < turn_chance else STRAIGHT_ROUTES
            route_id = rng.choice(routes_selected)
            vtype = rng.choice(VEHICLE_TYPES, p=VEHICLE_TYPE_WEIGHTS)
            car_row = _get_car_row(route_id=route_id, car_i=car_i, step=step, vtype=vtype)

            print(car_row, file=routes_file)

        print("</routes>", file=routes_file)
        routes_file.flush()
