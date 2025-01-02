import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.cache import VenuePreparationCache
from app.model import DeliveryTimeModel
from app.schemas import OrderRequest, PredictionResponse

# Set up logging
logger = logging.getLogger("inference_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


MODEL_PATH = os.getenv("MODEL_PATH", "/app/model_artifact/_model_artifact.json")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_DATA_PATH = os.getenv("CACHE_PATH", "/app/data/_venue_preparation.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model
    try:
        app.state.model = DeliveryTimeModel(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    # Initialize cache
    try:
        app.state.cache = VenuePreparationCache(host=REDIS_HOST, port=REDIS_PORT)
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise e

    # Load cache data
    try:
        app.state.cache.load_cache(CACHE_DATA_PATH)
        logger.info(f"Cache loaded successfully from {CACHE_DATA_PATH}")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        raise e

    yield
    logger.info("Shutting down inference service.")


# Initialize FastAPI
app = FastAPI(title="Delivery Time Prediction API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict_delivery_time(order: OrderRequest):
    try:
        # Extract features
        hour = order.time_received.hour  # Follows training preprocess function
        is_retail = order.is_retail
        avg_prep_time = app.state.cache.get_avg_preparation_time(order.venue_id)
        logger.info(
            f"Fetched avg_preparation_time: {avg_prep_time} for venue_id: {order.venue_id}"
        )

        # Prepare feature dataframe
        feature_dict = {
            "is_retail": [is_retail],
            "avg_preparation_time": [avg_prep_time],
            "hour_of_day": [hour],
        }
        df_features = pd.DataFrame(feature_dict)

        # Predict
        delivery_duration = app.state.model.predict(df_features)
        logger.info(f"Predicted delivery_duration: {delivery_duration}")

        return PredictionResponse(delivery_duration=delivery_duration)

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
