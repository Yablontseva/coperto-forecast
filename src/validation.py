import pandas as pd
from sklearn.metrics import mean_absolute_error


def time_split(
    df: pd.DataFrame,
    valid_days: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит данные по времени.

    Последние valid_days наблюдений используются
    для валидации, более ранние — для обучения.
    """

    validation_start = (
        df["date"].max()
        - pd.Timedelta(days=valid_days - 1)
    )

    train_data = df[
        df["date"] < validation_start
    ].copy()

    valid_data = df[
        df["date"] >= validation_start
    ].copy()

    return train_data, valid_data


def calculate_metrics(
    y_true: pd.Series,
    y_pred
) -> dict[str, float]:
    """
    Рассчитывает MAE и MAPE.

    MAPE рассчитывается только для наблюдений,
    где фактическое значение больше нуля.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    non_zero_mask = y_true != 0

    mape = (
        abs(
            (
                y_true[non_zero_mask]
                - y_pred[non_zero_mask]
            )
            / y_true[non_zero_mask]
        ).mean()
        * 100
    )

    return {
        "MAE": mae,
        "MAPE": mape
    }