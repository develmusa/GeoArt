from pydantic import BaseModel, ValidationError
from geopy.geocoders import Nominatim

class LocationPoint(BaseModel):
    latitude: float
    longitude: float


def address_to_coordinates(address: str) -> LocationPoint:
    geolocator = Nominatim(user_agent="geoart")
    location = geolocator.geocode(address)
    if location is None:
        raise ValueError(f"Could not find location for address: {address}")
    
    return LocationPoint(latitude=location.latitude, longitude=location.longitude)