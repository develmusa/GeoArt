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
def create_colormap_preview(cmap_name, width=200, height=15):
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

# Function to handle checkbox selection
def select_colormap(cmap_name, key_prefix):
    """Handle checkbox selection to ensure only one is selected at a time"""
    # Set all checkboxes to False
    for cmap in st.session_state:
        if cmap.startswith(f"{key_prefix}_checkbox_"):
            st.session_state[cmap] = False
    
    # Set the selected checkbox to True
    st.session_state[f"{key_prefix}_checkbox_{cmap_name}"] = True
    
    # Update the selected colormap
    st.session_state[f"{key_prefix}_selected"] = cmap_name

def display_colormap_grid(colormap_categories, selected_category=None, key_prefix="grid"):
    """
    Display a list of colormap previews organized by category with checkbox selection.
    
    Parameters:
    - colormap_categories: Dictionary mapping category names to lists of colormap names
    - selected_category: Currently selected category (optional)
    - key_prefix: Prefix for session state keys to avoid conflicts
    
    Returns:
    - Selected colormap name
    """
    # Initialize session state for selected colormap if not exists
    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = "viridis"
    
    # If no category is selected, show all categories
    if selected_category is None:
        categories_to_show = list(colormap_categories.keys())
    else:
        categories_to_show = [selected_category]
    
    # Category descriptions
    category_descriptions = {
        'Sequential': "Colors that progress from low to high. Good for continuous data.",
        'Diverging': "Colors that diverge from a central point. Good for data with a meaningful center point.",
        'Qualitative': "Distinct colors without implied order. Good for categorical data."
    }
    
    # Create tabs for each category
    tabs = st.tabs(categories_to_show)
    
    for i, category in enumerate(categories_to_show):
        with tabs[i]:
            # Display category description
            st.info(category_descriptions.get(category, ""))
            st.caption(f"{len(colormap_categories[category])} colormaps")
            
            # Get all colormap names in this category
            cmap_names = colormap_categories[category]
            
            # Display colormaps in a list with everything on the same horizontal line
            for cmap_name in cmap_names:
                # Initialize checkbox state if not exists
                checkbox_key = f"{key_prefix}_checkbox_{cmap_name}"
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = (cmap_name == st.session_state[f"{key_prefix}_selected"])
                
                # Create a container for each row
                with st.container():
                    # Use a single row with custom HTML for better alignment
                    col1, col2, col3 = st.columns([0.5, 1.5, 6])
                    
                    with col1:
                        # Checkbox for selection
                        is_selected = st.checkbox(
                            "",
                            value=st.session_state[checkbox_key],
                            key=checkbox_key,
                            on_change=select_colormap,
                            args=(cmap_name, key_prefix),
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        # Display colormap name with vertical alignment
                        st.markdown(f"<div style='margin-top: 5px;'><b>{cmap_name}</b></div>", unsafe_allow_html=True)
                    
                    with col3:
                        # Display colormap preview
                        fig = create_colormap_preview(cmap_name)
                        st.pyplot(fig, bbox_inches='tight', pad_inches=0, use_container_width=True)
    
    return st.session_state[f"{key_prefix}_selected"]

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
    
    # Get all colormap categories
    all_categories = get_colormap_categories()
    
    # Main area
    
    # Create an expander for the colormap selection
    with st.expander("Select a Colormap", expanded=False):
        # Display colormap grid with all categories
        selected_from_grid = display_colormap_grid(
            all_categories,
            key_prefix=f"{key_prefix}_grid"
        )
        
        # Update selected colormap if a new one was selected from the grid
        if selected_from_grid:
            st.session_state[f"{key_prefix}_selected"] = selected_from_grid
    
    # Get the currently selected colormap
    selected_cmap = st.session_state[f"{key_prefix}_selected"]
    
    # Options for the selected colormap
    reverse = st.checkbox("Reverse Colormap", value=st.session_state[f"{key_prefix}_reverse"], key=f"{key_prefix}_reverse_checkbox")
    st.session_state[f"{key_prefix}_reverse"] = reverse
    
    # Get the display colormap name with _r suffix if reversed
    display_cmap = f"{selected_cmap}_r" if reverse else selected_cmap
    
    st.subheader(f"Current Selection: {display_cmap}")
    # Show preview of selected colormap with reverse option
    fig = create_colormap_preview(display_cmap)
    st.pyplot(fig, bbox_inches='tight', pad_inches=0, use_container_width=True)

    # Return the selected configuration
    return {
        'colormap': selected_cmap,
        'reverse': reverse
    }

class SessionStateManager(BaseModel):
    # User input fields
    location: str = Field(default="Kopenhagen")
    start_date: datetime.date = Field(default=datetime.date(2023, 1, 1))
    style: str = Field(default="afmhot")
    min_temp: float | None = Field(default=None)
    max_temp: float | None = Field(default=None)
    
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
def get_image(weather_data: WeatherData, color_map: str, min_temp: float = None, max_temp: float = None) -> Image:
    try:
        return create_image(weather_data.hourly.to_dataframe(), color_map, min_temp, max_temp)
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

form_col1, form_col2 = st.columns(2)
address = form_col1.text_input(label="Location", key="location")
start_date = form_col2.date_input("Start Date", min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366), key="start_date")

# Update computed fields
session.end_date = start_date.replace(year=start_date.year + 1)
session.location_coordinates = get_location_coordinates(address)
session.weather_data = get_weather_data(session.location_coordinates, session.start_date, session.end_date)

# Get temperature range for the data to use as hints
df = session.weather_data.hourly.to_dataframe()
min_data_temp = float(df['temperature'].min())
max_data_temp = float(df['temperature'].max())

# Temperature range controls
st.header("Temperature Range Mapping")
st.info(f"Data temperature range: {min_data_temp:.1f}°C to {max_data_temp:.1f}°C")
st.caption("Adjust min/max temperature values to control the color mapping range and enhance visualization.")

# Initialize session state for tracking if min/max temp have been set
if 'min_temp_set' not in st.session_state:
    st.session_state['min_temp_set'] = False
    # Initialize with data min on first load
    if session.min_temp is None:
        session.min_temp = min_data_temp

if 'max_temp_set' not in st.session_state:
    st.session_state['max_temp_set'] = False
    # Initialize with data max on first load
    if session.max_temp is None:
        session.max_temp = max_data_temp
# Add explanations for contrast operations
st.markdown("""
**Contrast Operations:**
- **Contrast Stretching**: Maps the full temperature range to the full color range, showing the complete data distribution.
- **Enhance Contrast**: Narrows the mapping range by 10% from each end, enhancing visibility of mid-range temperatures.
""")

# Add buttons for common operations
col1, col2 = st.columns(2)
with col1:
    # Add a button for contrast stretching (reset to data range)
    if st.button("Apply Contrast Stretching", help="Maps the full temperature range to the full color range. This maximizes the use of all available colors across the entire data range."):
        # Set the temperature values to the data range
        session.min_temp = min_data_temp
        session.max_temp = max_data_temp
        # Reset the flags
        st.session_state['min_temp_set'] = False
        st.session_state['max_temp_set'] = False
        # Force rerun with the new values
        st.rerun()
        
with col2:
    # Add a button for enhancing contrast with a narrower range
    if st.button("Enhance Contrast", help="Narrows the mapping range by 10% from each end. This increases contrast in the middle range where most temperature values typically fall."):
        # Calculate a narrower range (e.g., 10% in from each end)
        range_size = max_data_temp - min_data_temp
        enhanced_min = min_data_temp + (range_size * 0.1)
        enhanced_max = max_data_temp - (range_size * 0.1)
        
        # Set the temperature values to the enhanced range
        session.min_temp = enhanced_min
        session.max_temp = enhanced_max
        # Set the flags
        st.session_state['min_temp_set'] = True
        st.session_state['max_temp_set'] = True
        # Force rerun with the new values
        st.rerun()
        st.rerun()

# Store current values for validation
current_min = float(session.min_temp)
current_max = float(session.max_temp)

temp_col1, temp_col2 = st.columns(2)
with temp_col1:
    min_temp = st.number_input(
        "Min Temperature (°C)",
        value=current_min,
        step=1.0,
        help="Minimum temperature for color mapping."
    )
    
    # Update session state
    if min_temp <= current_max:
        session.min_temp = min_temp
        if not st.session_state['min_temp_set'] and min_temp != min_data_temp:
            st.session_state['min_temp_set'] = True
    else:
        # If min > max, show error and don't update
        st.error("Min temperature cannot be greater than max temperature")
        # Reset to previous valid value
        session.min_temp = current_min

with temp_col2:
    # Ensure max_temp is at least min_temp
    max_temp = st.number_input(
        "Max Temperature (°C)",
        value=max(current_max, min_temp),
        min_value=min_temp,  # Enforce min_temp as the minimum allowed value
        step=1.0,
        help="Maximum temperature for color mapping."
    )
    
    # Update session state
    session.max_temp = max_temp
    if not st.session_state['max_temp_set'] and max_temp != max_data_temp:
        st.session_state['max_temp_set'] = True

# Add a note about the validation
st.caption("Note: Maximum temperature must be greater than or equal to minimum temperature.")

def get_image_wrapper():
    # Apply colormap with reverse option if selected
    cmap = session.style
    session.image = get_image(session.weather_data, cmap, session.min_temp, session.max_temp)

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

# Create a color legend
cmap_name = session.style
fig, ax = plt.subplots(figsize=(10, 0.5))
plt.subplots_adjust(left=0.05, right=0.95)

# Get the actual min/max values used for normalization
min_val = session.min_temp if session.min_temp is not None else df['temperature'].min()
max_val = session.max_temp if session.max_temp is not None else df['temperature'].max()

# Create a color bar
norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
# Make sure we're using the correct colormap name (handle _r suffix)
try:
    colormap = mpl.colormaps[cmap_name]
except KeyError:
    # If the colormap with _r suffix doesn't exist, use the base colormap and reverse it
    base_cmap = cmap_name.replace('_r', '')
    colormap = mpl.colormaps[base_cmap].reversed()

cb = mpl.colorbar.ColorbarBase(ax, cmap=colormap, norm=norm, orientation='horizontal')
cb.set_label('Temperature (°C)')
st.pyplot(fig)

# Add explanation of color mapping
if st.session_state.get('min_temp_set', False) or st.session_state.get('max_temp_set', False):
    st.info(f"""
    **Custom Contrast Mapping Applied:**
    - Full data range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C
    - Current mapping range: {min_val:.1f}°C to {max_val:.1f}°C
    
    Values outside this range will be clipped to the min/max colors, enhancing contrast within the selected range.
    """)
else:
    st.caption("Full contrast stretching applied (using entire data range). Use the temperature controls above to enhance specific ranges.")

col1, col2, col3 = st.columns(3)

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