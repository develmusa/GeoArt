import pandas as pd
from pydantic import BaseModel 
from typing import List
from datetime import datetime
import httpx
from geoart.data_utils import normalize_dataframe_to_bytes
import geoart.geolocation as geolocation

class HourlyData(BaseModel):
    time: List[datetime]
    temperature_2m: List[float]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'time': self.time,
            'temperature': self.temperature_2m
        })
    
    def to_dataframe_byte_normalized(self) -> pd.DataFrame:
        df = self.to_dataframe()
        temperatur_df = df[['temperature']]
        df["temperature"] = normalize_dataframe_to_bytes(temperatur_df)
        return df
    

# df = df.set_index("timestamp")
# df_minute = df.resample("T").interpolate()
# print(df_minute)

# df = df.set_index("timestamp")
# df_minute = df.resample("T").ffill()  # Forward fill hourly values
# print(df_minute)
    


class WeatherData(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    timezone_abbreviation: str
    utc_offset_seconds: int
    hourly: HourlyData

class WeatherDataError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

# Function to fetch weather data
async def fetch_weather_data(location_point: geolocation.Coordinates, start_date: str, end_date: str) -> WeatherData:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location_point.latitude,
        "longitude": location_point.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "auto"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Raise for HTTP errors
            data = response.json()
            # Parse and validate the response using Pydantic
            weather_data = WeatherData(**data)
            return weather_data
        except httpx.HTTPStatusError as e:
            print("HTTP Error:", e)
            raise WeatherDataError(f"Failed to fetch weather data: {e}")
        except Exception as e:
            print("Unexpected Error:", e)
            raise WeatherDataError(f"An unexpected error occurred: {e}")