import streamlit as st
import datetime
import matplotlib as mpl
from streamlit_geoart.processor import ProcessData, generate_year_temp_art 
from pydantic import BaseModel




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
    process_data: ProcessData = None
    location: str = "Kopenhagen"
    start_date: datetime.date = datetime.date(2023, 1, 1) 
    style: str = "berlin"
if not st.session_state:
    st.session_state.update(session_state_user_data())


form_col1, form_col2= st.columns(2)
address = form_col1.text_input(label="Location", key="location")
start_date = form_col2.date_input("Start Date", min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366), key="start_date")

def generate_data():
    data = generate_year_temp_art(
        location_address=st.session_state.location,
        color_map=st.session_state.style,
        start_date=st.session_state.start_date,
    )
    st.session_state.process_data = data

generate_data()

        
process_data: ProcessData = st.session_state.process_data

option = st.selectbox(
    "Select a colormap?",
    mpl.colormaps,
    key="style",
    on_change=generate_data,
)

st.image(process_data.image.get_image())

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