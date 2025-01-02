import pandas as pd
import xgboost as xgb


class DeliveryTimeModel:
    """
    A trained XGBRegressor() model saved using booster and ready for faster inference
    utilizing DMatrix data format
    """

    def __init__(self, model_path: str = "_model_artifact.json"):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, features: pd.DataFrame) -> float:
        # Convert DataFrame to DMatrix
        dmatrix = xgb.DMatrix(features)
        prediction = self.model.predict(dmatrix)
        return float(prediction[0])
