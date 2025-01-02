import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train(x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_hat))

    print(f"Train RMSE: {rmse}")
    return model


def evaluate(model: XGBRegressor, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_hat = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_hat))

    print(f"Test RMSE: {rmse}")


def preprocess(venue_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    orders_df["hour_of_day"] = pd.to_datetime(orders_df["time_received"]).dt.hour
    merged_df = venue_df.merge(orders_df, on="venue_id")

    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty.")

    return merged_df[["is_retail", "avg_preparation_time", "hour_of_day"]]


def main():
    venue_df = pd.read_csv("venue_preparation.csv")
    orders_df = pd.read_csv("orders_data.csv")

    x = preprocess(venue_df, orders_df)
    y = orders_df["delivery_duration"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    model = train(x_train, y_train)
    evaluate(model, x_test, y_test)
    model.save_model("model_artifact.json")


if __name__ == '__main__':
    main()
