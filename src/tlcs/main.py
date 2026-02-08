import csv
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
from torch import Tensor

from tlcs.agent import Agent
from tlcs.baseline import FixedTimeAgent
from tlcs.constants import TESTING_SETTINGS_FILE, TRAINING_SETTINGS_FILE
from tlcs.env import Environment, EnvStats
from tlcs.episode import Record, run_episode
from tlcs.logger import add_file_handler, get_logger
from tlcs.memory import Memory, Sample
from tlcs.plots import save_data_and_plot
from tlcs.settings import load_testing_settings, load_training_settings

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


class TrainingStats(TypedDict):
    """Aggregated statistics collected during training episodes."""

    sum_neg_reward: list[float]
    cumulative_wait: list[int]
    avg_queue_length: list[float]


class CheckpointState(TypedDict):
    """State captured in a training checkpoint."""

    episode: int
    epsilon: float
    model_state_dict: dict[str, Tensor]
    target_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, Tensor]
    memory_samples: list[Sample]
    training_stats: TrainingStats
    rng_state: dict[str, object]


class TestingStats(TypedDict):
    """Statistics collected during a testing episode."""

    reward: list[float]
    queue_length: list[int]


class EvaluationRow(TypedDict):
    """Per-seed evaluation metrics."""

    seed: int
    total_reward: float
    avg_queue_length: float


def add_experience_to_memory(memory: Memory, history: list[Record]) -> None:
    """Add transitions from an episode history to replay memory.

    Each pair of consecutive records is converted into a (s, a, r, s') sample.

    Args:
        memory: Replay memory buffer used by the agent.
        history: Ordered list of records from a single episode.
    """
    for i in range(len(history) - 1):
        sample = Sample(
            state=history[i].state,
            action=history[i].action,
            reward=history[i].reward,
            next_state=history[i + 1].state,
        )
        memory.add_sample(sample)


def update_training_stats(
    episode_history: list[Record],
    env_stats: list[EnvStats],
    max_steps: int,
    training_stats: TrainingStats,
) -> TrainingStats:
    """Update cumulative training statistics with metrics from one episode.

    The function tracks cumulative negative reward, cumulative waiting time and average queue
    length.

    Args:
        episode_history: Sequence of records produced during the episode.
        env_stats: Per-step environment statistics for the episode.
        max_steps: Maximum number of steps per episode (used for averaging).
        training_stats: Dictionary of training statistics to be updated.

    Returns:
        The updated training statistics.
    """
    # accumulate only negative rewards for clearer trend
    sum_neg_reward = sum(record.reward for record in episode_history if record.reward < 0)
    training_stats["sum_neg_reward"].append(sum_neg_reward)

    sum_queue_length = sum(stats.queue_length for stats in env_stats)
    avg_queue_length = round(sum_queue_length / max_steps, 1)
    training_stats["avg_queue_length"].append(avg_queue_length)

    # 1 car in queue for 1 step == 1 second of waiting time
    training_stats["cumulative_wait"].append(sum_queue_length)

    return training_stats


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value to apply.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict[str, object]:
    rng_state: dict[str, object] = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _set_rng_state(rng_state: dict[str, object]) -> None:
    if "random" in rng_state:
        random.setstate(cast("tuple[Any, ...]", rng_state["random"]))
    if "numpy" in rng_state:
        np.random.set_state(cast("tuple[Any, ...]", rng_state["numpy"]))  # noqa: NPY002
    if "torch" in rng_state:
        torch.set_rng_state(cast("Tensor", rng_state["torch"]))
    if torch.cuda.is_available() and "torch_cuda" in rng_state:
        torch.cuda.set_rng_state_all(cast("Iterable[Tensor]", rng_state["torch_cuda"]))


def save_checkpoint(
    out_path: Path,
    episode: int,
    agent: Agent,
    memory: Memory,
    training_stats: TrainingStats,
) -> Path:
    """Save a training checkpoint and return its path."""
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"

    state: CheckpointState = {
        "episode": episode,
        "epsilon": agent.epsilon,
        "model_state_dict": agent.model.state_dict(),
        "target_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.model.optimizer_state_dict(),
        "memory_samples": list(memory.samples),
        "training_stats": training_stats,
        "rng_state": _get_rng_state(),
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    agent: Agent,
    memory: Memory,
) -> tuple[int, TrainingStats]:
    """Load a training checkpoint and return next episode index + stats."""
    state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    agent.model.load_state_dict(state["model_state_dict"])
    agent.target_model.load_state_dict(state["target_state_dict"])
    agent.model.load_optimizer_state_dict(state["optimizer_state_dict"])
    agent.set_epsilon(state["epsilon"])

    memory.samples.clear()
    for sample in state["memory_samples"]:
        memory.add_sample(sample)

    if "rng_state" in state:
        _set_rng_state(state["rng_state"])

    next_episode = int(state["episode"]) + 1
    training_stats = state["training_stats"]
    return next_episode, training_stats


def training_session(  # noqa: PLR0915
    settings_file: Path,
    out_path: Path,
    seed_override: int | None = None,
    resume_from: Path | None = None,
) -> None:
    """Run a full training session and save the trained model and statistics.

    Args:
        settings_file: Path to the training settings file.
        out_path: Directory where model, settings and plots will be saved.
        seed_override: Optional seed value to override the settings file.
        resume_from: Optional checkpoint path to resume training from.
    """
    settings = load_training_settings(settings_file)
    if seed_override is not None:
        settings = settings.model_copy(update={"seed": seed_override})

    if settings.seed is not None and resume_from is None:
        set_global_seed(settings.seed)

    out_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(out_path / "train.log")

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)
    agent = Agent(settings=settings)

    timestamp_start = datetime.now()
    tot_episodes = settings.total_episodes

    training_stats: TrainingStats = {
        "sum_neg_reward": [],
        "cumulative_wait": [],
        "avg_queue_length": [],
    }
    start_episode = 0

    if resume_from is not None:
        start_episode, training_stats = load_checkpoint(
            checkpoint_path=resume_from,
            agent=agent,
            memory=memory,
        )

    metrics_csv_path = out_path / "metrics.csv"
    metrics_jsonl_path = out_path / "metrics.jsonl"

    csv_mode = "a" if start_episode > 0 and metrics_csv_path.exists() else "w"
    jsonl_mode = "a" if start_episode > 0 and metrics_jsonl_path.exists() else "w"

    with metrics_csv_path.open(
        csv_mode,
        encoding="utf-8",
        newline="",
    ) as csv_file, metrics_jsonl_path.open(
        jsonl_mode,
        encoding="utf-8",
    ) as jsonl_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "epsilon",
                "sum_neg_reward",
                "cumulative_wait",
                "avg_queue_length",
            ],
        )
        if csv_mode == "w":
            csv_writer.writeheader()

        for episode in range(start_episode, tot_episodes):
            logger.info(f"Episode {episode + 1} of {tot_episodes}")

            new_epsilon = round(1.0 - (episode / tot_episodes), 2)
            agent.set_epsilon(new_epsilon)

            env = Environment(
                n_cars_generated=settings.n_cars_generated,
                max_steps=settings.max_steps,
                yellow_duration=settings.yellow_duration,
                green_duration=settings.green_duration,
                turn_chance=settings.turn_chance,
                gui=settings.gui,
                sumocfg_file=settings.sumocfg_file,
            )

            episode_seed = episode
            if settings.seed is not None:
                episode_seed = settings.seed + episode

            episode_history, env_stats = run_episode(env=env, agent=agent, seed=episode_seed)

            add_experience_to_memory(memory=memory, history=episode_history)

            for _ in range(settings.training_epochs):
                agent.replay(
                    memory=memory,
                    gamma=settings.gamma,
                    batch_size=settings.batch_size,
                )

            training_stats = update_training_stats(
                episode_history=episode_history,
                env_stats=env_stats,
                max_steps=settings.max_steps,
                training_stats=training_stats,
            )

            last_neg_reward = training_stats["sum_neg_reward"][-1]
            last_cumulative_wait = training_stats["cumulative_wait"][-1]
            last_avg_queue_length = training_stats["avg_queue_length"][-1]

            logger.info(f"\tEpsilon: {agent.epsilon}")
            logger.info(f"\tReward: {last_neg_reward}")
            logger.info(f"\tCumulative wait: {last_cumulative_wait}")
            logger.info(f"\tAvg queue: {last_avg_queue_length}")

            row = {
                "episode": episode + 1,
                "epsilon": agent.epsilon,
                "sum_neg_reward": last_neg_reward,
                "cumulative_wait": last_cumulative_wait,
                "avg_queue_length": last_avg_queue_length,
            }
            csv_writer.writerow(row)
            jsonl_file.write(json.dumps(row) + "\n")
            csv_file.flush()
            jsonl_file.flush()

            if (episode + 1) % settings.checkpoint_interval == 0:
                checkpoint_path = save_checkpoint(
                    out_path=out_path,
                    episode=episode,
                    agent=agent,
                    memory=memory,
                    training_stats=training_stats,
                )
                logger.info(f"\tCheckpoint saved: {checkpoint_path}")

    agent.save_model(out_path)

    logger.info(f"Start time: {timestamp_start}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Session info saved at: {out_path}")

    copyfile(src=settings_file, dst=out_path / TRAINING_SETTINGS_FILE)

    save_data_and_plot(
        data=training_stats["sum_neg_reward"],
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["cumulative_wait"],
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["avg_queue_length"],
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
        out_folder=out_path,
    )


def testing_session(settings_file: Path, model_path: Path, test_name: str) -> None:
    """Load a trained agent and run a single testing episode, saving plots and settings.

    Args:
        settings_file: Path to the testing settings file.
        model_path: Path to the directory containing the trained model and training settings.
        test_name: Name of the subdirectory where testing outputs will be saved.
    """
    settings = load_testing_settings(settings_file)

    test_path = model_path / test_name
    test_path.mkdir(parents=True, exist_ok=True)

    agent = Agent(
        settings=load_training_settings(model_path / TRAINING_SETTINGS_FILE),
        epsilon=0,
        model_path=model_path,
    )

    env = Environment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        gui=settings.gui,
        sumocfg_file=settings.sumocfg_file,
    )

    episode_history, env_stats = run_episode(
        env=env,
        agent=agent,
        seed=settings.episode_seed,
    )

    testing_stats: TestingStats = {
        "reward": [record.reward for record in episode_history],
        "queue_length": [stats.queue_length for stats in env_stats],
    }

    copyfile(src=settings_file, dst=test_path / TESTING_SETTINGS_FILE)

    save_data_and_plot(
        data=testing_stats["reward"],
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
        out_folder=test_path,
    )
    save_data_and_plot(
        data=testing_stats["queue_length"],
        filename="queue",
        xlabel="Step",
        ylabel="Queue length (vehicles)",
        out_folder=test_path,
    )

    logger.info(f"Testing results saved at: {test_path}")


def evaluation_session(
    settings_file: Path,
    model_path: Path,
    eval_name: str,
    seeds: list[int] | None = None,
) -> None:
    """Evaluate a trained agent across multiple seeds and report aggregates.

    Args:
        settings_file: Path to the testing settings file.
        model_path: Path to the directory containing the trained model.
        eval_name: Name of the evaluation output folder.
        seeds: Optional list of episode seeds to evaluate.
    """
    settings = load_testing_settings(settings_file)

    if not seeds:
        seeds = [settings.episode_seed + i for i in range(5)]

    eval_path = model_path / eval_name
    eval_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(eval_path / "eval.log")

    agent = Agent(
        settings=load_training_settings(model_path / TRAINING_SETTINGS_FILE),
        epsilon=0,
        model_path=model_path,
    )

    rows: list[EvaluationRow] = []

    for seed in seeds:
        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        episode_history, env_stats = run_episode(env=env, agent=agent, seed=seed)

        total_reward = float(sum(record.reward for record in episode_history))
        avg_queue_length = float(
            round(sum(stats.queue_length for stats in env_stats) / settings.max_steps, 3)
        )

        rows.append(
            {
                "seed": seed,
                "total_reward": total_reward,
                "avg_queue_length": avg_queue_length,
            }
        )

        logger.info(f"Eval seed {seed}: reward={total_reward}, avg_queue={avg_queue_length}")

    rewards = [row["total_reward"] for row in rows]
    queues = [row["avg_queue_length"] for row in rows]

    summary = {
        "seeds": seeds,
        "reward_mean": float(statistics.mean(rewards)),
        "reward_std": float(statistics.pstdev(rewards)),
        "avg_queue_mean": float(statistics.mean(queues)),
        "avg_queue_std": float(statistics.pstdev(queues)),
    }

    with (eval_path / "eval_metrics.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["seed", "total_reward", "avg_queue_length"])
        writer.writeheader()
        writer.writerows(rows)

    with (eval_path / "eval_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    copyfile(src=settings_file, dst=eval_path / TESTING_SETTINGS_FILE)
    logger.info(f"Evaluation results saved at: {eval_path}")


def baseline_session(
    settings_file: Path,
    eval_name: str,
    seeds: list[int] | None = None,
) -> None:
    """Evaluate a fixed-time baseline across multiple seeds.

    Args:
        settings_file: Path to the testing settings file.
        eval_name: Name of the evaluation output folder.
        seeds: Optional list of episode seeds to evaluate.
    """
    settings = load_testing_settings(settings_file)

    if not seeds:
        seeds = [settings.episode_seed + i for i in range(5)]

    eval_path = Path("model") / eval_name
    eval_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(eval_path / "eval.log")

    agent = FixedTimeAgent()

    rows: list[EvaluationRow] = []

    for seed in seeds:
        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        episode_history, env_stats = run_episode(env=env, agent=agent, seed=seed)

        total_reward = float(sum(record.reward for record in episode_history))
        avg_queue_length = float(
            round(sum(stats.queue_length for stats in env_stats) / settings.max_steps, 3)
        )

        rows.append(
            {
                "seed": seed,
                "total_reward": total_reward,
                "avg_queue_length": avg_queue_length,
            }
        )

        logger.info(f"Baseline seed {seed}: reward={total_reward}, avg_queue={avg_queue_length}")

    rewards = [row["total_reward"] for row in rows]
    queues = [row["avg_queue_length"] for row in rows]

    summary = {
        "seeds": seeds,
        "reward_mean": float(statistics.mean(rewards)),
        "reward_std": float(statistics.pstdev(rewards)),
        "avg_queue_mean": float(statistics.mean(queues)),
        "avg_queue_std": float(statistics.pstdev(queues)),
    }

    with (eval_path / "eval_metrics.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["seed", "total_reward", "avg_queue_length"])
        writer.writeheader()
        writer.writerows(rows)

    with (eval_path / "eval_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    copyfile(src=settings_file, dst=eval_path / TESTING_SETTINGS_FILE)
    logger.info(f"Baseline results saved at: {eval_path}")
