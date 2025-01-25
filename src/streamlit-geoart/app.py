import streamlit as st

import datetime
from geoart.processor import generate_year_temp_art, ProcessData


st.set_page_config(
    page_title="GeoArt", page_icon="🗺️", initial_sidebar_state="collapsed"
)

st.title('GeoArt')

st.write("""
    # One Year Temperatures
    GeoArt is a tool for creating and sharing geospatial art.
    """)

coordinates = None

form = st.form(key="form_settings")
address = form.text_input(label="Location", key="location")
start_date = form.date_input("Start Date", datetime.date(2019, 7, 6))
form_submit = form.form_submit_button(label="Submit")

if form_submit:
    with st.status("Creating Arte...", expanded=True) as status:
        def status_progress_callback(message: str):
            status.update(label=message, state="running")

        def status_error_callback(message: str):
            status.update(label=message, state="error")

        def status_success_callback(message: str, data: ProcessData):
            status.update(label=message, state="complete", expanded=False)
            status.write(data)

        
        generate_year_temp_art(
            location_address=address,
            start_date=start_date,
            success_callback=status_success_callback,
            progress_callback=status_progress_callback,
            error_callback=status_error_callback
        )
            

