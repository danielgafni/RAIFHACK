import typing

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1


def deviation_metric_one_sample(y_true: typing.Union[float, int], y_pred: typing.Union[float, int]) -> float:
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= -4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2  # type: ignore
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2  # type: ignore
    else:
        return 9


def deviation_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()  # type: ignore


def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.median(np.abs(y_pred - y_true) / y_true)  # type: ignore


def metrics_stat(y_true: np.ndarray, y_pred: np.ndarray) -> typing.Dict[str, float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric(y_true, y_pred)
    return {
        "mape": mape,
        "mdape": mdape,
        "rmse": rmse,
        "r2": r2,
        "raif_metric": raif_metric,
    }


EPS = 1e-8

assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])) <= EPS
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([0.9, 1.8, 2.7, 3.6, 4.5])) <= EPS
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1.1, 2.2, 3.3, 4.4, 5.5])) <= EPS
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1.15, 2.3, 3.45, 4.6, 5.75])) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1.3, 2.6, 3.9, 5.2, 6.5])) - 1) <= EPS
assert (
    np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([0.7, 1.4, 2.1, 2.8, 3.5])) - 1 * NEGATIVE_WEIGHT)
    <= EPS
)
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50])) - 9) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 0])) - 9 * NEGATIVE_WEIGHT) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1, 2.2, 3.3, 5, 50])) - 85 / 45) <= EPS
