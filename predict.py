import argparse
from pathlib import Path

import pandas as pd
import joblib

from src.data import load_processed_data
from src.features import make_features


FEATURE_COLUMNS = [
    "day_of_week",
    "is_weekend",
    "month",
    "week_of_year",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
]


DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "processed"
    / "restaurant_cleaned.csv"
)

MODEL_PATH = (
    Path(__file__).parent
    / "data"
    / "processed"
    / "restaurant_model.joblib"
)


def make_forecast(
    df: pd.DataFrame,
    model,
    forecast_date: str,
    restaurant_id: int,
    horizon: int = 7
) -> pd.DataFrame:
    """
    Строит рекурсивный прогноз на несколько дней вперёд.
    """

    forecast_date = pd.to_datetime(
        forecast_date
    )

    restaurant_data = df[
        df["restaurant_id"] == restaurant_id
    ].copy()

    if restaurant_data.empty:
        raise ValueError(
            f"Ресторан с ID {restaurant_id} не найден."
        )

    last_date = restaurant_data["date"].max()

    if forecast_date <= last_date:
        raise ValueError(
            "Дата прогноза должна быть позже "
            f"последней даты данных: {last_date.date()}"
        )

    if forecast_date > last_date + pd.Timedelta(days=1):
        raise ValueError(
            "Для этого датасета дата прогноза должна "
            "следовать непосредственно за последней "
            f"датой данных: {(last_date + pd.Timedelta(days=1)).date()}"
        )

    history = restaurant_data[
        ["date", "inside_covers"]
    ].copy()

    forecast_dates = pd.date_range(
        start=forecast_date,
        periods=horizon,
        freq="D"
    )

    predictions = []

    for current_date in forecast_dates:

        current_row = pd.DataFrame({
            "date": [current_date],
            "inside_covers": [None]
        })

        temp_data = pd.concat(
            [history, current_row],
            ignore_index=True
        )

        temp_features = make_features(
            temp_data,
            target_col="inside_covers"
        )

        current_features = temp_features[
            temp_features["date"] == current_date
        ][FEATURE_COLUMNS]

        prediction = model.predict(
            current_features
        )[0]

        prediction = max(
            0,
            prediction
        )

        predictions.append(
            {
                "date": current_date,
                "restaurant_id": restaurant_id,
                "predicted_guests": round(
                    prediction,
                    2
                )
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame({
                    "date": [current_date],
                    "inside_covers": [prediction]
                })
            ],
            ignore_index=True
        )

    return pd.DataFrame(predictions)


def main() -> None:
    """
    Точка входа CLI.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Прогноз количества гостей "
            "на следующие 7 дней."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Дата начала прогноза в формате YYYY-MM-DD"
    )

    parser.add_argument(
        "--restaurant",
        required=True,
        type=int,
        help="ID ресторана"
    )

    args = parser.parse_args()

    df = load_processed_data(
        DATA_PATH
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    forecast = make_forecast(
        df=df,
        model=model,
        forecast_date=args.date,
        restaurant_id=args.restaurant,
        horizon=7
    )

    print()
    print("Прогноз количества гостей:")
    print()

    print(
        forecast.to_string(
            index=False
        )
    )

    output_path = (
        Path(__file__).parent
        / "data"
        / "processed"
        / "forecast.csv"
    )
    
    forecast.to_csv(
        output_path,
        index=False
    )
    
    print()
    print("Прогноз сохранён:")
    print(output_path)


if __name__ == "__main__":
    main()