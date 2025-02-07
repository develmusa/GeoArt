import asyncio
import datetime
from typing import Optional

import streamlit as st
from pydantic import BaseModel 

import geoart.geolocation as geolocation
import geoart.weather_data as weather_data
from geoart.geolocation import Coordinates
from geoart.image import Image, create_image
from geoart.weather_data import WeatherData


class ProcessData(BaseModel):
    location_address: str
    location_coordinates: Optional[Coordinates] = None
    weather_data: Optional[WeatherData] = None
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    image: Optional[Image] = None
    # image: Optional[np.ndarray] = None  # To hold serialized NumPy array    image_array: Optional[np.ndarray] = None

@st.cache_data
def generate_year_temp_art(
    location_address: str,
    color_map: str,
    start_date: datetime.date,
) -> ProcessData:  # Define the exact return type if possible
    process_data = ProcessData(location_address=location_address, start_date=start_date)
    process_data.end_date = start_date.replace(year=start_date.year + 1)

    async def async_wrapper() -> ProcessData:
        try:
            process_data.location_coordinates = geolocation.address_to_coordinates(address=location_address)
            process_data.weather_data = await weather_data.fetch_weather_data(
                location_point=process_data.location_coordinates,
                start_date=process_data.start_date,
                end_date=process_data.end_date
            )
            process_data.image = create_image(process_data.weather_data.hourly.to_dataframe(), color_map)
            return process_data

        except geolocation.AddressNotFoundError:
            st.error(f"Could not find coordinates for address: {location_address}")
            return process_data

        except weather_data.WeatherDataError:
            st.error(f"Could not fetch weather data for {location_address}")
            return process_data

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            return process_data

    return asyncio.run(async_wrapper())
