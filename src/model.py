from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def create_model() -> Pipeline:
    """
    Создаёт модель Random Forest для прогнозирования количества гостей.
    """

    model = Pipeline([
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return model


def train_model(
    X_train,
    y_train
) -> Pipeline:
    """
    Обучает Random Forest на обучающей выборке.
    """

    model = create_model()

    model.fit(X_train, y_train)

    return model