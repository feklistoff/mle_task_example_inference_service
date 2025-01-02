import pandas as pd
import redis


class VenuePreparationCache:
    """
    A base class to load and read data using Redis
    """

    def __init__(self, host: str = "redis", port: int = 6379):
        self.client = redis.Redis(host=host, port=port, db=0)

    def get_avg_preparation_time(self, venue_id: str) -> float:
        value = self.client.get(venue_id)
        if value is None:
            raise ValueError(f"Venue ID {venue_id} not found in cache.")
        return float(value)

    def load_cache(self, data_path: str):
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            self.client.set(row["venue_id"], row["avg_preparation_time"])
