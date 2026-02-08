import numpy as np

from tlcs.model import Model


def test_model_forward_shapes() -> None:
    model = Model(
        num_layers=1,
        width=16,
        learning_rate=0.001,
        input_dim=4,
        output_dim=2,
    )

    state = np.zeros(4, dtype=np.float32)
    pred = model.predict_one(state)
    assert pred.shape == (1, 2)

    batch = np.zeros((3, 4), dtype=np.float32)
    preds = model.predict_batch(batch)
    assert preds.shape == (3, 2)
