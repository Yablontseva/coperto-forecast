import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Добавляет календарные признаки для прогнозирования количества гостей.
    """

    result = df.copy()

    result[date_col] = pd.to_datetime(result[date_col])

    result["day_of_week"] = result[date_col].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["month"] = result[date_col].dt.month
    result["week_of_year"] = (
        result[date_col]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    calendar = USFederalHolidayCalendar()

    holidays = calendar.holidays(
        start=result[date_col].min(),
        end=result[date_col].max()
    )

    result["is_holiday"] = result[date_col].isin(holidays).astype(int)

    return result

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "inside_covers",
    lags: tuple[int, ...] = (1, 7, 14)
) -> pd.DataFrame:
    """
    Добавляет лаговые признаки для целевой переменной.
    """

    result = df.copy()

    result = result.sort_values("date").reset_index(drop=True)

    for lag in lags:
        result[f"lag_{lag}"] = result[target_col].shift(lag)

    return result

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "inside_covers",
    windows: tuple[int, ...] = (7, 28)
) -> pd.DataFrame:
    """
    Добавляет скользящие средние целевой переменной
    без использования значения текущего дня.
    """

    result = df.copy()

    result = result.sort_values("date").reset_index(drop=True)

    for window in windows:
        result[f"rolling_mean_{window}"] = (
            result[target_col]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    return result

def make_features(
    df: pd.DataFrame,
    target_col: str = "inside_covers"
) -> pd.DataFrame:
    """
    Создаёт полный набор признаков для прогнозирования.
    """

    result = add_calendar_features(df)

    result = add_lag_features(
        result,
        target_col=target_col
    )

    result = add_rolling_features(
        result,
        target_col=target_col
    )

    return result