# src/geoart/tests/test_geolocation.py

import pytest
from unittest.mock import patch
from geopy.exc import GeocoderServiceError

from geoart.geolocation import LocationPoint, address_to_goordinates


def test_address_to_goordinates_valid():
    address = "1600 Amphitheatre Parkway, Mountain View, CA"
    expected_location = LocationPoint(latitude=37.4221, longitude=-122.0841)
    
    with patch('geopy.geocoders.Nominatim.geocode') as mock_geocode:
        mock_geocode.return_value.latitude = expected_location.latitude
        mock_geocode.return_value.longitude = expected_location.longitude
        
        location = address_to_goordinates(address)
        assert location == expected_location

def test_address_to_goordinates_invalid():
    address = "Some Invalid Address"
    
    with patch('geopy.geocoders.Nominatim.geocode', return_value=None):
        with pytest.raises(ValueError, match=f"Could not find location for address: {address}"):
            address_to_goordinates(address)

def test_address_to_goordinates_service_error():
    address = "1600 Amphitheatre Parkway, Mountain View, CA"
    
    with patch('geopy.geocoders.Nominatim.geocode', side_effect=GeocoderServiceError):
        with pytest.raises(GeocoderServiceError):
            address_to_goordinates(address)
