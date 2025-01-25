import streamlit as st
import time
import geoart.geolocation as geolocation
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl
import pandas as pd
st.set_page_config(
    page_title="GeoArt", page_icon="🗺️", initial_sidebar_state="collapsed"
)

st.title('GeoArt')

st.write("""
    # Year Temperatures
    GeoArt is a tool for creating and sharing geospatial art.
    """)

coordinates = None


# if not st.session_state:
form = st.form(key="form_settings")
address = form.text_input(label="Location" , key="location")
form = form.form_submit_button(label="Submit")
if form:
    with st.status("Creating Arte...", expanded=True) as status:
        st.write("Get Location Coordinates...")
        coordinates = geolocation.address_to_coordinates(address=address)
        status.update(
            # label=f"Download complete!", state="complete" 
            label=f"Download complete!", state="complete", expanded=False
        )
    # st.map( coordinates, zoom=15)

    st.map(data=pd.DataFrame([coordinates.model_dump()]))