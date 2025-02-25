import asyncio
import time
import streamlit as st
import datetime
import matplotlib as mpl
from geoart import geolocation
from geoart import weather_data
from geoart.image import Image, create_image
from geoart.weather_data import WeatherData
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Tuple, Optional
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# Cache the colormap preview to improve performance
@st.cache_data
def create_colormap_preview(cmap_name, width=200, height=30):
    """Create a preview image for a colormap without borders"""
    # Create figure with tight layout and no padding
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding
    
    # Create gradient
    gradient = np.linspace(0, 1, width)
    gradient = np.vstack((gradient, gradient))
    
    # Display gradient
    ax.imshow(gradient, aspect='auto', cmap=cmap_name)
    
    # Remove axes and borders
    ax.axis('off')
    fig.patch.set_alpha(0)  # Make figure background transparent
    
    # Set tight layout with no padding
    plt.tight_layout(pad=0)
    
    return fig

# Create a sample data preview to show how the colormap would look on temperature data
@st.cache_data
def create_sample_data_preview(cmap_name, width=200, height=100):
    """Create a preview of how the colormap would look on sample temperature data"""
    # Create sample temperature data (sine wave pattern)
    x = np.linspace(0, 2*np.pi, width)
    y = np.linspace(0, 2*np.pi, height)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # Create figure with tight layout and no padding
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding
    
    # Display the sample data with the colormap
    im = ax.imshow(Z, aspect='auto', cmap=cmap_name)
    
    # Remove axes and borders
    ax.axis('off')
    fig.patch.set_alpha(0)  # Make figure background transparent
    
    # Set tight layout with no padding
    plt.tight_layout(pad=0)
    
    return fig

def get_colormap_categories():
    """Get colormaps organized by category"""
    # Define common colormaps by category
    sequential_cmaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]
    
    diverging_cmaps = [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    ]
    
    qualitative_cmaps = [
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
        'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ]
    
    return {
        'Sequential': sequential_cmaps,
        'Diverging': diverging_cmaps,
        'Qualitative': qualitative_cmaps
    }

def display_colormap_grid(colormap_categories, selected_category=None, key_prefix="grid"):
    """
    Display a grid of colormap previews organized by category.
    
    Parameters:
    - colormap_categories: Dictionary mapping category names to lists of colormap names
    - selected_category: Currently selected category (optional)
    - key_prefix: Prefix for session state keys to avoid conflicts
    
    Returns:
    - Selected colormap name
    """
    # If no category is selected, show all categories
    if selected_category is None:
        categories_to_show = list(colormap_categories.keys())
    else:
        categories_to_show = [selected_category]
    
    selected_cmap = None
    
    # Create tabs for each category
    tabs = st.tabs(categories_to_show)
    
    for i, category in enumerate(categories_to_show):
        with tabs[i]:
            st.caption(f"{len(colormap_categories[category])} colormaps")
            
            # Create a grid layout with 4 columns
            cols = st.columns(4)
            
            # Display colormaps in the grid
            for j, cmap_name in enumerate(colormap_categories[category]):
                with cols[j % 4]:
                    # Create a container for the colormap
                    with st.container():
                        st.caption(cmap_name)
                        
                        # Create the colormap preview
                        fig = create_colormap_preview(cmap_name)
                        
                        # Make the preview clickable
                        if st.button("Select", key=f"{key_prefix}_{category}_{cmap_name}"):
                            selected_cmap = cmap_name
                            
                        # Display the preview
                        st.pyplot(fig, bbox_inches='tight', pad_inches=0, use_container_width=True)
    
    return selected_cmap

def colormap_selector(key_prefix="colormap", display_mode="grid"):
    """
    Enhanced colormap selector that returns the selected colormap configuration.
    
    Parameters:
    - key_prefix: Prefix for session state keys to avoid conflicts
    - display_mode: 'grid' for grid display, 'single' for single preview
    
    Returns:
    - Dictionary with colormap configuration
    """
    # Initialize session state for colormap selection if not exists
    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = "viridis"
    
    if f"{key_prefix}_reverse" not in st.session_state:
        st.session_state[f"{key_prefix}_reverse"] = False
    
    if f"{key_prefix}_map_type" not in st.session_state:
        st.session_state[f"{key_prefix}_map_type"] = "Continuous"
    
    if f"{key_prefix}_n_colors" not in st.session_state:
        st.session_state[f"{key_prefix}_n_colors"] = 5
    
    # Sidebar for categories and filters
    with st.sidebar:
        st.subheader("Filter Colormaps")
        show_sequential = st.checkbox("Sequential", value=True, key=f"{key_prefix}_show_sequential")
        show_diverging = st.checkbox("Diverging", value=True, key=f"{key_prefix}_show_diverging")
        show_qualitative = st.checkbox("Qualitative", value=True, key=f"{key_prefix}_show_qualitative")
        
        # Search box
        search_term = st.text_input("Search colormaps", key=f"{key_prefix}_search").lower()

    # Get colormap categories
    all_categories = get_colormap_categories()
    
    # Filter categories based on checkboxes
    filtered_categories = {}
    if show_sequential:
        filtered_categories['Sequential'] = all_categories['Sequential']
    if show_diverging:
        filtered_categories['Diverging'] = all_categories['Diverging']
    if show_qualitative:
        filtered_categories['Qualitative'] = all_categories['Qualitative']
    
    # Filter by search term
    if search_term:
        for category in filtered_categories:
            filtered_categories[category] = [
                cm for cm in filtered_categories[category]
                if search_term in cm.lower()
            ]
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Select a Colormap")
        
        if display_mode == "grid":
            # Display colormap grid
            selected_from_grid = display_colormap_grid(
                filtered_categories,
                key_prefix=f"{key_prefix}_grid"
            )
            
            # Update selected colormap if a new one was selected from the grid
            if selected_from_grid:
                st.session_state[f"{key_prefix}_selected"] = selected_from_grid
        
        # Flatten the list of available colormaps
        available_cmaps = []
        for category in filtered_categories:
            available_cmaps.extend(filtered_categories[category])
        
        # If no colormaps are available after filtering, show a message
        if not available_cmaps:
            st.warning("No colormaps match your filters. Try adjusting your search or filter settings.")
            # Use a default colormap
            available_cmaps = ["viridis"]
        
        # Current selection (as a backup and for precise selection)
        selected_cmap = st.selectbox(
            "Or select from dropdown",
            available_cmaps,
            index=available_cmaps.index(st.session_state[f"{key_prefix}_selected"]) if st.session_state[f"{key_prefix}_selected"] in available_cmaps else 0,
            key=f"{key_prefix}_selectbox"
        )
        
        # Update the session state
        st.session_state[f"{key_prefix}_selected"] = selected_cmap
        
        st.subheader("Current Selection")
        # Show preview of selected colormap
        fig = create_colormap_preview(selected_cmap)
        st.pyplot(fig, bbox_inches='tight', pad_inches=0, use_container_width=True)
        
        # Show sample data preview
        st.caption("Sample data preview")
        sample_fig = create_sample_data_preview(selected_cmap)
        st.pyplot(sample_fig, bbox_inches='tight', pad_inches=0, use_container_width=True)
        
        # Options for the selected colormap
        reverse = st.checkbox("Reverse Colormap", value=st.session_state[f"{key_prefix}_reverse"], key=f"{key_prefix}_reverse_checkbox")
        st.session_state[f"{key_prefix}_reverse"] = reverse
        
        # Discrete vs Continuous
        map_type = st.radio("Colormap Type", ["Continuous", "Discrete"], index=0 if st.session_state[f"{key_prefix}_map_type"] == "Continuous" else 1, key=f"{key_prefix}_map_type_radio")
        st.session_state[f"{key_prefix}_map_type"] = map_type
        
        n_colors = 5
        if map_type == "Discrete":
            n_colors = st.slider("Number of colors", 2, 20, st.session_state[f"{key_prefix}_n_colors"], key=f"{key_prefix}_n_colors_slider")
            st.session_state[f"{key_prefix}_n_colors"] = n_colors

    with col2:
        st.subheader("Options")
        
        # Range selector
        st.write("Value Range")
        min_val = st.number_input("Min", value=0.0, key=f"{key_prefix}_min_val")
        max_val = st.number_input("Max", value=1.0, key=f"{key_prefix}_max_val")
        
        # Information about colormap categories
        st.subheader("Colormap Types")
        
        st.markdown("""
        **Sequential**: Colors that progress from low to high. Good for continuous data.
        
        **Diverging**: Colors that diverge from a central point. Good for data with a meaningful center point.
        
        **Qualitative**: Distinct colors without implied order. Good for categorical data.
        """)

    # Return the selected configuration
    return {
        'colormap': selected_cmap,
        'reverse': reverse,
        'type': map_type,
        'n_colors': n_colors if map_type == "Discrete" else None,
        'range': (min_val, max_val)
    }

class SessionStateManager(BaseModel):
    # User input fields
    location: str = Field(default="Kopenhagen")
    start_date: datetime.date = Field(default=datetime.date(2023, 1, 1))
    style: str = Field(default="afmhot")
    
    # Computed fields
    end_date: datetime.date | None = Field(default=None)
    location_coordinates: geolocation.Coordinates | None = Field(default=None)
    weather_data: WeatherData | None = Field(default=None)
    image: Image | None = Field(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to sync with st.session_state"""
        super().__setattr__(name, value)
        if hasattr(st, 'session_state'):  # Check if we're in a Streamlit context
            setattr(st.session_state, name, value)

    @classmethod
    def from_session_state(cls) -> 'SessionStateManager':
        """Create a SessionState instance from current st.session_state and ensure defaults are set"""
        instance = cls(**{
            field: getattr(st.session_state, field, default.default)
            for field, default in cls.model_fields.items()
        })
        
        # Ensure st.session_state contains all fields with their values
        for field, value in instance.model_dump().items():
            if not hasattr(st.session_state, field):
                setattr(st.session_state, field, value)
        
        return instance

    

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
    page_title="GeoArt", page_icon="🗺️", initial_sidebar_state="expanded"
)

st.title('GeoArt')

st.write("""
    # One Year Temperature
    This app generates a temperature map for a given location of a year.
    The map is generated using the OpenWeatherMap API.
    """)
# Initialize or get existing session state
session = SessionStateManager.from_session_state()

form_col1, form_col2= st.columns(2)
address = form_col1.text_input(label="Location", key="location")
start_date = form_col2.date_input("Start Date", min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366), key="start_date")

# Update computed fields
session.end_date = start_date.replace(year=start_date.year + 1)
session.location_coordinates = get_location_coordinates(address)
session.weather_data = get_weather_data(session.location_coordinates, session.start_date, session.end_date)

def get_image_wrapper():
    # Apply colormap with reverse option if selected
    cmap = session.style
    if "_r" in cmap:
        cmap = cmap.replace("_r", "")
    session.image = get_image(session.weather_data, cmap)

# Use the enhanced colormap selector
st.header("Colormap Selection")
colormap_config = colormap_selector(key_prefix="main_colormap", display_mode="grid")

# Update the style in session state
if colormap_config['reverse']:
    session.style = f"{colormap_config['colormap']}_r"
else:
    session.style = colormap_config['colormap']

# Generate the image with the selected colormap
get_image_wrapper()

# Display the generated image
st.header("Generated Temperature Visualization")
st.image(session.image.get_image())

col1, col2, col3 = st.columns(3)
# Convert hourly weather data to a DataFrame
df = session.weather_data.hourly.to_dataframe()

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