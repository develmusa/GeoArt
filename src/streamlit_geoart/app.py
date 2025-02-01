import json
import time
import pandas as pd
import streamlit as st
import datetime
import pendulum
import matplotlib as mpl
from streamlit_geoart.processor import ProcessData, generate_year_temp_art
from pydantic import BaseModel

class session_state_user_data(BaseModel):
    process_data: ProcessData = None
    location: str = "Kopenhagen"
    start_date: datetime.date = datetime.date(2023, 1, 1) 
    style: str = "berlin"


st.set_page_config(
    page_title="GeoArt", page_icon="🗺️", initial_sidebar_state="collapsed"
)

st.title('GeoArt')

st.write("""
    # One Year Temperature
    This app generates a temperature map for a given location of a year.
    The map is generated using the OpenWeatherMap API.
    """)

if not st.session_state:
    st.session_state.process_data = None

# with open("./src/streamlit-geoart/examples.json", "r", encoding="utf8") as f:
    # EXAMPLES = json.load(f)
# st.session_state.update(EXAMPLES["example"])

if not st.session_state.process_data:
    st.session_state.update(session_state_user_data())

# for item in st.session_state.items():
    # st.write(item)




form_col1, form_col2= st.columns(2)
address = form_col1.text_input(label="Location", key="location")
start_date = form_col2.date_input("Start Date", min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366), key="start_date")


with st.status("Creating Arte...", expanded=True) as status:
    def status_progress_callback(message: str):
        status.update(label=message, state="running", expanded=True)

    def status_error_callback(message: str):
        status.update(label=message, state="error")

    def status_success_callback(message: str, data: ProcessData):
        status.update(label=message, state="complete", expanded=True)
        st.session_state.process_data = data
    
    generate_year_temp_art(
        location_address=st.session_state.location,
        color_map=st.session_state.style,
        start_date=st.session_state.start_date,
        success_callback=status_success_callback,
        progress_callback=status_progress_callback,
        error_callback=status_error_callback
    )
def generate_data():
    ...
    generate_year_temp_art(
        location_address=st.session_state.location,
        color_map=st.session_state.style,
        start_date=st.session_state.start_date,
        success_callback=status_success_callback,
        progress_callback=status_progress_callback,
        error_callback=status_error_callback
    )


data: ProcessData = st.session_state.process_data

        
process_data: ProcessData = st.session_state.process_data

option = st.selectbox(
    "Select a colormap?",
    mpl.colormaps,
    key="style",
    on_change=generate_data,
)

st.image(process_data.image.get_image())
# print(process_data.model_dump_json)
col1, col2, col3 = st.columns(3)
    # Convert hourly weather data to a DataFrame
df = process_data.weather_data.hourly.to_dataframe()

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