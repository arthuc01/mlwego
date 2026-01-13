import numpy as np

from mlwego.evaluation.metrics import get_metric


def test_metric_direction() -> None:
    metric_fn, direction = get_metric("rmse")
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 1.0, 2.0])
    score = metric_fn(y_true, y_pred)
    assert score == 0.0
    assert direction == "max"
