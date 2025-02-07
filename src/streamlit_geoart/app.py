import asyncio
import streamlit as st
import datetime
import matplotlib as mpl
from geoart import geolocation
from geoart import weather_data
from geoart.image import Image, create_image
from geoart.weather_data import WeatherData
from pydantic import BaseModel

@st.cache_data
def get_location_coordinates(location_str: str) -> geolocation.Coordinates:
    try:
        return geolocation.address_to_coordinates(address=location_str)
    except geolocation.AddressNotFoundError:
        st.error(f"Could not find coordinates for address: {location_str}")
        st.stop()
    except Exception as e:
        #Todo: Add logging and error tracking do not expose internal errors to the user
        st.error("An unkown error occurred while fetching location coordinates")
        st.stop()
        

@st.cache_data(hash_funcs={geolocation.Coordinates: lambda c: (c.latitude, c.longitude)})
def get_weather_data(location: geolocation.Coordinates, start_date: datetime.date, end_date: datetime.date) -> WeatherData:
    try:
        async def async_wrapper() -> WeatherData:
            return await weather_data.fetch_weather_data(location_point=location, start_date=start_date, end_date=end_date)
        return asyncio.run(async_wrapper())
    except weather_data.WeatherDataError:
        st.error(f"Could not fetch weather data for {location}, {start_date} to {end_date}")
        st.stop()
    except Exception as e:
        #Todo: Add logging and error tracking do not expose internal errors to the user
        st.error("An unkown error occurred while fetching weather data")
        st.stop()

@st.cache_data(hash_funcs={WeatherData: lambda w: (w.latitude, w.longitude, hash(tuple(w.hourly.temperature_2m)))})
def get_image(weather_data: WeatherData, color_map: str) -> Image:
    try:
        return create_image(weather_data.hourly.to_dataframe(), color_map)
    except Exception as e:  
        #Todo: Add logging and error tracking do not expose internal errors to the user
        st.error("An unkown error occurred while generating image")
        st.stop()


st.set_page_config(
    page_title="GeoArt", page_icon="🗺️", initial_sidebar_state="collapsed"
)

st.title('GeoArt')

st.write("""
    # One Year Temperature
    This app generates a temperature map for a given location of a year.
    The map is generated using the OpenWeatherMap API.
    """)

class session_state_user_data(BaseModel):
    location: str = "Kopenhagen"
    start_date: datetime.date = datetime.date(2023, 1, 1) 
    style: str = "berlin"
if not st.session_state:
    st.session_state.update(session_state_user_data())


form_col1, form_col2= st.columns(2)
address = form_col1.text_input(label="Location", key="location")
start_date = form_col2.date_input("Start Date", min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366), key="start_date")
st.session_state.end_date = start_date.replace(year=start_date.year + 1)

st.session_state.location_coordinates = get_location_coordinates(address)
st.session_state.weather_data = get_weather_data(st.session_state.location_coordinates, st.session_state.start_date, st.session_state.end_date)

def get_image_wrapper():
    st.session_state.image = get_image(st.session_state.weather_data, st.session_state.style)

get_image_wrapper()

option = st.selectbox(
    "Select a colormap?",
    mpl.colormaps,
    key="style",
    on_change=get_image_wrapper,
)

st.image(st.session_state.image.get_image(), caption="Temperature Map")

col1, col2, col3 = st.columns(3)
    # Convert hourly weather data to a DataFrame
df = st.session_state.weather_data.hourly.to_dataframe()

# Find indices of maximum and minimum temperatures
max_temp_index = df['temperature'].idxmax()
min_temp_index = df['temperature'].idxmin()

with col1:
    st.markdown("##### Max. Temperature")
    max_time_str = df.iloc[max_temp_index]['time'].strftime('%Y-%m-%d %H:%M:%S')
    max_temp_str = f"{df.iloc[max_temp_index]['temperature']} °C"
    st.metric(label=max_time_str, value=max_temp_str, border=False)

with col2:
    st.markdown("##### Min. Temperature")
    min_time_str = df.iloc[min_temp_index]['time'].strftime('%Y-%m-%d %H:%M:%S')
    min_temp_str = f"{df.iloc[min_temp_index]['temperature']} °C"
    st.metric(label=min_time_str, value=min_temp_str)

with col3:
    st.markdown("##### Mean Temperature")
    mean_temp_str = f"{df['temperature'].mean().round()} °C"
    st.metric(label="Mean", label_visibility="hidden", value=mean_temp_str)

    # map_data = pd.DataFrame([st.session_state.process_data.location_coordinates.model_dump()])
    # st.map(data=map_data, zoom=10, use_container_width=True)