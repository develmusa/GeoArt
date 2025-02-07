from pydantic import BaseModel 
from geopy.geocoders import Nominatim

class Coordinates(BaseModel):
    latitude: float
    longitude: float
    
    # class Config:
    #     frozen = True  # Make the class immutable
    
    # def __hash__(self) -> int:
    #     return hash((self.latitude, self.longitude))

class GeoLocationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class AddressNotFoundError(GeoLocationError):
    def __init__(self, message: str):
        super().__init__(message)


def address_to_coordinates(address: str) -> Coordinates:
    try:
        geolocator = Nominatim(user_agent="geoart")
        location = geolocator.geocode(address)
        if location is None:
            raise AddressNotFoundError(f"Could not find coordinates for address: {address}")
        return Coordinates(latitude=location.latitude, longitude=location.longitude)
    except Exception as e:
        raise GeoLocationError(f"An error occurred while geocoding the address: {e}")