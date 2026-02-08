import pytest

from tlcs.settings import TrainingSettings


def test_training_settings_rejects_invalid_memory_bounds() -> None:
    with pytest.raises(ValueError):
        TrainingSettings(
            gui=False,
            total_episodes=1,
            max_steps=10,
            n_cars_generated=10,
            green_duration=5,
            yellow_duration=2,
            turn_chance=0.2,
            num_layers=1,
            width_layers=8,
            batch_size=4,
            learning_rate=0.001,
            training_epochs=1,
            memory_size_min=10,
            memory_size_max=10,
            gamma=0.9,
            sumocfg_file="intersection/sumo_config.sumocfg",
            use_double_dqn=True,
            target_update_interval=5,
            grad_clip_norm=1.0,
            seed=123,
            checkpoint_interval=5,
        )
