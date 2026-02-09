import numpy as np

from tlcs.memory import Memory, Sample


def test_memory_sampling_respects_min_size() -> None:
    memory = Memory(size_max=10, size_min=3)

    assert memory.get_samples(2) == []

    for i in range(2):
        memory.add_sample(
            Sample(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1], dtype=np.float32),
            )
        )

    assert memory.get_samples(2) == []

    memory.add_sample(
        Sample(
            state=np.array([2], dtype=np.float32),
            action=1,
            reward=2.0,
            next_state=np.array([3], dtype=np.float32),
        )
    )

    samples = memory.get_samples(2)
    assert len(samples) == 2
