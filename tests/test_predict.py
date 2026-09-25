import pandas as pd
import joblib

from src.data import load_processed_data
from predict import make_forecast


DATA_PATH = "data/processed/restaurant_cleaned.csv"
MODEL_PATH = "data/processed/restaurant_model.joblib"


def test_make_forecast_returns_seven_days():
    """
    Проверяет, что функция возвращает прогноз на 7 дней.
    """

    df = load_processed_data(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    forecast = make_forecast(
        df=df,
        model=model,
        forecast_date="2019-07-01",
        restaurant_id=1,
        horizon=7
    )

    assert len(forecast) == 7


def test_forecast_dates_are_consecutive():
    """
    Проверяет последовательность дат прогноза.
    """

    df = load_processed_data(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    forecast = make_forecast(
        df=df,
        model=model,
        forecast_date="2019-07-01",
        restaurant_id=1,
        horizon=7
    )

    expected_dates = pd.date_range(
        "2019-07-01",
        periods=7,
        freq="D"
    )

    assert forecast["date"].tolist() == expected_dates.tolist()


def test_forecast_is_non_negative():
    """
    Количество гостей не может быть отрицательным.
    """

    df = load_processed_data(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    forecast = make_forecast(
        df=df,
        model=model,
        forecast_date="2019-07-01",
        restaurant_id=1,
        horizon=7
    )

    assert (forecast["predicted_guests"] >= 0).all()


def test_unknown_restaurant_raises_error():
    """
    Проверяет ошибку для неизвестного ресторана.
    """

    df = load_processed_data(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    try:
        make_forecast(
            df=df,
            model=model,
            forecast_date="2019-07-01",
            restaurant_id=999,
            horizon=7
        )
    except ValueError:
        assert True
    else:
        assert False