import time
from dataclasses import dataclass
from typing import Protocol

from numpy.typing import NDArray

from tlcs.env import Environment, EnvStats


class ActionAgent(Protocol):
    """Protocol for agents that choose actions from states."""

    def choose_action(self, state: NDArray) -> int:
        """Return an action index for the given state."""


@dataclass
class Record:
    """Single time-step transition experienced during an episode.

    Attributes:
        state: Environment state observed before taking the action.
        action: Action chosen by the agent.
        reward: Reward obtained after executing the action.
    """

    state: NDArray
    action: int
    reward: float


def run_episode(
    env: Environment,
    agent: ActionAgent,
    seed: int,
    queue_penalty_weight: float = 0.0,
    max_queue_penalty_weight: float = 0.0,
) -> tuple[list[Record], list[EnvStats]]:
    """Runs one episode and returns per-step records and environment statistics.

    Args:
        env: Environment to interact with.
        agent: Agent used to select actions from states.
        seed: Seed used to generate the route file for this episode.
        queue_penalty_weight: Weight for average queue penalty in reward.
        max_queue_penalty_weight: Weight for max-queue penalty in reward.

    Returns:
        A tuple (history, env_stats) where:
            history is the list of per-step records.
            env_stats is the list of environment statistics for each executed action.
    """
    env.generate_routefile(seed=seed)
    time.sleep(0.1)  # Ensure route file is written before SUMO starts

    previous_total_wait = 0.0
    history: list[Record] = []
    env_stats: list[EnvStats] = []

    env.activate()

    while not env.is_over():
        state = env.get_state()
        action = agent.choose_action(state)

        action_stats = env.execute(action)
        env_stats.extend(action_stats)

        current_total_wait = env.get_cumulated_waiting_time()
        reward = previous_total_wait - current_total_wait
        if action_stats:
            avg_queue = float(sum(s.queue_length for s in action_stats)) / len(action_stats)
            max_queue = max(s.max_queue for s in action_stats)
            reward -= queue_penalty_weight * avg_queue
            reward -= max_queue_penalty_weight * float(max_queue)
        previous_total_wait = current_total_wait

        record = Record(state=state, action=action, reward=reward)
        history.append(record)

    env.deactivate()

    return history, env_stats
