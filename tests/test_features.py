import pandas as pd

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features
)


def test_add_calendar_features():
    """Проверяет создание календарных признаков."""

    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-06"
        ])
    })

    result = add_calendar_features(df)

    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert "month" in result.columns
    assert "week_of_year" in result.columns
    assert "is_holiday" in result.columns

    # 2024-01-01 — понедельник
    assert result.loc[0, "day_of_week"] == 0

    # 2024-01-06 — суббота
    assert result.loc[1, "day_of_week"] == 5
    assert result.loc[1, "is_weekend"] == 1


def test_add_lag_features():
    """Проверяет корректность лаговых признаков."""

    df = pd.DataFrame({
        "date": pd.date_range(
            "2024-01-01",
            periods=5,
            freq="D"
        ),
        "inside_covers": [100, 110, 120, 130, 140]
    })

    result = add_lag_features(df)

    # Для первого дня истории ещё нет
    assert pd.isna(result.loc[0, "lag_1"])

    # Для второго дня lag_1 должен быть равен значению первого дня
    assert result.loc[1, "lag_1"] == 100


def test_add_rolling_features():
    """Проверяет, что rolling не использует текущий день."""

    df = pd.DataFrame({
        "date": pd.date_range(
            "2024-01-01",
            periods=8,
            freq="D"
        ),
        "inside_covers": [
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170
        ]
    })

    result = add_rolling_features(
        df,
        windows=(7,)
    )

    # Для седьмого дня ещё недостаточно семи предыдущих наблюдений
    assert pd.isna(
        result.loc[6, "rolling_mean_7"]
    )

    # Для восьмого дня используются значения первых семи дней, без текущего значения 170
    assert result.loc[7, "rolling_mean_7"] == 130