from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "inside_covers",
    "restaurant_id",
]


def load_processed_data(
    path: Union[str, Path]
) -> pd.DataFrame:
    """
    Загружает обработанный датасет и проверяет его структуру.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл с данными не найден: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "В датасете отсутствуют обязательные столбцы: "
            f"{missing_columns}"
        )

    df["restaurant_id"] = (
        df["restaurant_id"]
        .astype("int64")
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["restaurant_id", "date"]
    ).reset_index(drop=True)

    return df