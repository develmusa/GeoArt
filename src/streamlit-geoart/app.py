import pandas as pd
import streamlit as st

import datetime
from geoart.processor import generate_year_temp_art, ProcessData
import matplotlib as mpl


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





form = st.form(key="form_settings")
address = form.text_input(label="Location", key="location")
start_date = form.date_input("Start Date", datetime.date(2024, 1, 1), min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366))
form_submit = form.form_submit_button(label="Submit")

if form_submit:
    with st.status("Creating Arte...", expanded=True) as status:
        def status_progress_callback(message: str):
            status.update(label=message, state="running", expanded=True)

        def status_error_callback(message: str):
            status.update(label=message, state="error")

        def status_success_callback(message: str, data: ProcessData):
            status.update(label=message, state="complete", expanded=True)
            st.session_state.process_data = data

            # status.write(data)

        
        generate_year_temp_art(
            location_address=address,
            color_map="magma",
            start_date=start_date,
            success_callback=status_success_callback,
            progress_callback=status_progress_callback,
            error_callback=status_error_callback
        )

    # st.write(st.session_state.process_data)
    # coordinates = st.session_state.process_data.coordinates
    # process_data: ProcessData = st.session_state.process_data
    data: ProcessData = st.session_state.process_data
    map_data = pd.DataFrame([st.session_state.process_data.location_coordinates.model_dump()])


            
    process_data: ProcessData = st.session_state.process_data

    option = st.selectbox(
        "Select a colormap?",
        mpl.colormaps,
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

    st.map(data=map_data, zoom=10, use_container_width=True)