import asyncio
import time
import streamlit as st
import datetime
import matplotlib as mpl
import pandas as pd
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
    
    # Simplified category descriptions
    category_descriptions = {
        'Sequential': "Colors that progress from low to high. Good for continuous data.",
        'Diverging': "Colors that diverge from a central point. Good for data with a meaningful center point.",
        'Qualitative': "Distinct colors without implied order. Good for categorical data."
    }
    
    # Create tabs for each category
    tabs = st.tabs(categories_to_show)
    
    for i, category in enumerate(categories_to_show):
        with tabs[i]:
            # Category descriptions and counts removed as per user request
            
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
                        # Checkbox for selection with proper label
                        is_selected = st.checkbox(
                            f"Select {cmap_name}",
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
                        plt.close(fig)  # Close figure to prevent memory leaks
    
    return st.session_state[f"{key_prefix}_selected"]

def colormap_selector(key_prefix="colormap", display_mode="grid", use_expander=True):
    """
    Enhanced colormap selector that returns the selected colormap configuration.
    
    Parameters:
    - key_prefix: Prefix for session state keys to avoid conflicts
    - display_mode: 'grid' for grid display, 'single' for single preview
    - use_expander: Whether to use an expander for the colormap grid (set to False when used inside another expander)
    
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
    
    # Get the currently selected colormap
    selected_cmap = st.session_state[f"{key_prefix}_selected"]
    
    # Show current selection at the top
    st.subheader(f"Current Selection: {selected_cmap}")
    
    # Get the display colormap name with _r suffix if reversed
    display_cmap = f"{selected_cmap}_r" if st.session_state[f"{key_prefix}_reverse"] else selected_cmap
    
    # Show preview of selected colormap with reverse option
    fig = create_colormap_preview(display_cmap)
    st.pyplot(fig, bbox_inches='tight', pad_inches=0, use_container_width=True)
    plt.close(fig)  # Close figure to prevent memory leaks
    
    # Options for the selected colormap after the plot
    reverse = st.checkbox("Reverse Colormap", value=st.session_state[f"{key_prefix}_reverse"], key=f"{key_prefix}_reverse_checkbox", help="Invert the colormap direction")
    st.session_state[f"{key_prefix}_reverse"] = reverse
    
    # If reverse option changed, rerun to update the preview
    if reverse != st.session_state[f"{key_prefix}_reverse"]:
        st.rerun()
    
    # Display colormap grid - either in an expander or directly
    if use_expander:
        with st.expander("Select a Different Colormap", expanded=False):
            # Colormap descriptions removed as per user request
            
            selected_from_grid = display_colormap_grid(
                all_categories,
                key_prefix=f"{key_prefix}_grid"
            )
            
            # Update selected colormap if a new one was selected from the grid
            if selected_from_grid:
                st.session_state[f"{key_prefix}_selected"] = selected_from_grid
    else:
        # Display directly without an expander
        st.subheader("Select a Different Colormap")
        
        # Colormap descriptions removed as per user request
        
        selected_from_grid = display_colormap_grid(
            all_categories,
            key_prefix=f"{key_prefix}_grid"
        )
        
        # Update selected colormap if a new one was selected from the grid
        if selected_from_grid:
            st.session_state[f"{key_prefix}_selected"] = selected_from_grid

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
    
    # Temperature range settings
    custom_range_min: float = Field(default=-20.0)
    custom_range_max: float = Field(default=60.0)
    min_temp_set: bool = Field(default=False)
    max_temp_set: bool = Field(default=False)
    
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
    
    def update_temp_range(self, min_temp: float, max_temp: float, data_min: float, data_max: float) -> None:
        """
        Update temperature range with validation
        
        Args:
            min_temp: New minimum temperature
            max_temp: New maximum temperature
            data_min: Minimum temperature in the data
            data_max: Maximum temperature in the data
        """
        # Validate min < max
        if min_temp >= max_temp:
            raise ValueError("Minimum temperature must be less than maximum temperature")
            
        # Update temperature values
        self.min_temp = min_temp
        self.max_temp = max_temp
        
        # Update flags based on data range
        self.min_temp_set = min_temp != data_min
        self.max_temp_set = max_temp != data_max
    

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
    page_title="Annual Temperature Progression", page_icon="🗺️", initial_sidebar_state="expanded"
)

# Function to get color for a temperature value
def get_color_for_temp(cmap_name, temp_value, min_temp, max_temp):
    """
    Get color for a temperature value from the specified colormap
    
    Args:
        cmap_name: Name of the colormap
        temp_value: Temperature value to get color for
        min_temp: Minimum temperature in the range
        max_temp: Maximum temperature in the range
        
    Returns:
        Hex color code for the temperature value
    """
    # Normalize the temperature value to [0, 1]
    if max_temp == min_temp:  # Avoid division by zero
        normalized_value = 0.5
    else:
        normalized_value = (temp_value - min_temp) / (max_temp - min_temp)
    
    # Clamp to [0, 1] range
    normalized_value = max(0, min(1, normalized_value))
    
    try:
        colormap = mpl.colormaps[cmap_name]
    except KeyError:
        # If the colormap with _r suffix doesn't exist, use the base colormap and reverse it
        base_cmap = cmap_name.replace('_r', '')
        colormap = mpl.colormaps[base_cmap].reversed()
    
    # Get the RGB color and convert to hex
    rgb_color = colormap(normalized_value)
    hex_color = mpl.colors.rgb2hex(rgb_color)
    return hex_color

# Add custom CSS to ensure consistent font size in matplotlib figures
# and to style the temperature input fields with colormap colors
st.markdown("""
<style>
    /* Ensure consistent font size in matplotlib figures */
    .stPlotlyChart, .stAgGrid, .stDataFrame {
        font-size: 14px !important;
    }
    
    /* Ensure matplotlib text is visible regardless of theme */
    .matplotlib-text {
        color: white !important;
    }
    
    /* We'll use a different approach with custom elements instead of trying to style the inputs directly */
    
    /* Number input styling removed as per user request */
</style>
""", unsafe_allow_html=True)
st.title('Annual Temperature Progression')

# Initialize or get existing session state
session = SessionStateManager.from_session_state()

# Get initial values for location and start date from session state
address = session.location
start_date = session.start_date

# Update computed fields
session.end_date = start_date.replace(year=start_date.year + 1)
session.location_coordinates = get_location_coordinates(address)
session.weather_data = get_weather_data(session.location_coordinates, session.start_date, session.end_date)

# Get temperature range for the data to use as hints
df = session.weather_data.hourly.to_dataframe()
min_data_temp = float(df['temperature'].min())
max_data_temp = float(df['temperature'].max())

# Define helper functions first
def get_temperature_range(min_temp, max_temp, df):
    """
    Calculate the applied temperature range without displaying a legend.
    
    Args:
        min_temp: Minimum temperature value (or None to use data min)
        max_temp: Maximum temperature value (or None to use data max)
        df: DataFrame containing temperature data
        
    Returns:
        tuple: (min_val, max_val) - The actual min/max values used for normalization
    """
    # Get the data range and the applied range
    data_min = float(df['temperature'].min())
    data_max = float(df['temperature'].max())
    applied_min = min_temp if min_temp is not None else data_min
    applied_max = max_temp if max_temp is not None else data_max
    
    return applied_min, applied_max

# Create a temperature legend/colorbar
def create_temperature_legend(cmap_name, min_temp, max_temp, width=600, height=15):
    """Create a temperature legend with the current colormap"""
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

# Initialize temperature range values if not set
if session.min_temp is None:
    session.min_temp = min_data_temp
    session.min_temp_set = False

if session.max_temp is None:
    session.max_temp = max_data_temp
    session.max_temp_set = False

# Initialize custom range limits if not set
data_range = max_data_temp - min_data_temp
if session.custom_range_min == -20.0 and session.custom_range_max == 60.0:
    # Only set if using default values
    session.custom_range_min = min_data_temp - (data_range * 0.5)
    session.custom_range_max = max_data_temp + (data_range * 0.5)

# Set default style if not already set
if not hasattr(session, 'style') or session.style is None:
    session.style = "viridis"

# Generate the visualization first (before customization controls)
def get_image_wrapper():
    cmap = session.style if hasattr(session, 'style') else "viridis"
    session.image = get_image(session.weather_data, cmap, session.min_temp, session.max_temp)

# Generate the image
get_image_wrapper()

# Display the visualization immediately after the title
st.image(session.image.get_image())

# Get temperature range values for legend
min_val, max_val = get_temperature_range(session.min_temp, session.max_temp, df)

# Add temperature legend
legend_fig = create_temperature_legend(
    session.style,
    min_val,
    max_val,
    width=600,
    height=15
)
st.pyplot(legend_fig, bbox_inches='tight', pad_inches=0, use_container_width=True)
plt.close(legend_fig)  # Close figure to prevent memory leaks

# Add legend ticks
legend_min = min_val
legend_max = max_val
legend_width = 100
num_ticks = 5
tick_temps = np.linspace(legend_min, legend_max, num_ticks)
tick_positions = [0, 25, 50, 75, 100]

tick_html = '<div class="legend-tick-container">'
for i, (pos, temp) in enumerate(zip(tick_positions, tick_temps)):
    tick_html += f'<div class="legend-tick-mark" style="left: {pos}%;"></div>'
    if i == 0:
        tick_html += f'<div class="legend-tick-label" style="left: {pos+2}%; text-align: left;">{temp:.1f}°C</div>'
    elif i == len(tick_positions) - 1:
        tick_html += f'<div class="legend-tick-label" style="left: {pos-2}%; text-align: right;">{temp:.1f}°C</div>'
    else:
        tick_html += f'<div class="legend-tick-label" style="left: {pos}%;">{temp:.1f}°C</div>'
tick_html += '</div>'
st.markdown(tick_html, unsafe_allow_html=True)

# Add input controls before the tabs
input_col1, input_col2 = st.columns(2)

with input_col1:
    new_address = st.text_input(
        label="Location",
        value=address,
        key="location",
        help="Enter an address (e.g., 'New York') or coordinates (e.g., '47.3769° N, 8.5417° E')"
    )

with input_col2:
    new_start_date = st.date_input("Start Date", value=start_date, min_value=datetime.date(1940, 1, 1), max_value=datetime.datetime.now()- datetime.timedelta(days=366), key="start_date")

# Check if inputs have changed and trigger a rerun if needed
if new_address != address or new_start_date != start_date:
    st.rerun()

# Create tabs for description, usage instructions, and technical details
description_tab, usage_tab, technical_tab = st.tabs(["Description", "How to Use", "Technical Details"])

with description_tab:
    st.markdown("""
    ### Annual Temperature Visualization
    
    This visualization shows temperature patterns throughout the year for a selected location. The heatmap displays:
    
    - **Horizontal axis**: Time of day (00:00 to 23:00)
    - **Vertical axis**: Day of the year (January to December)
    - **Color**: Temperature values
    
    Explore seasonal patterns, daily temperature cycles, and extreme weather events at your selected location.
    """)

with usage_tab:
    st.markdown("""
    ### How to Use This Visualization
    
    1. **View the visualization**: The main heatmap shows temperature patterns over time
    2. **Change location**: Enter a city or address in the "Location" field below
    3. **Select different year**: Use the "Start Date" picker to choose a different year
    4. **Customize colors**: Use the "Colormap Selection" in the sidebar to change the color scheme
    5. **Adjust temperature range**: Fine-tune the temperature scale using the controls in the sidebar
    
    The color legend shows the temperature scale used in the visualization.
    """)

with technical_tab:
    st.markdown("""
    ### Data and Visualization Details
    
    This heatmap is generated using hourly temperature data for a full year:
    
    - **Data source**: Historical weather data from OpenWeatherMap API
    - **Resolution**: Hourly temperature readings (8,760 data points per year)
    - **Color mapping**: Temperature values are mapped to colors using the selected colormap
    - **Temperature range**: By default, uses the actual min/max temperatures in the data
    
    ### Interpreting Patterns
    
    Different patterns in the visualization reveal different climate characteristics:
    
    - **Horizontal bands**: Show daily temperature cycles (warmer during day, cooler at night)
    - **Vertical progression**: Reveals seasonal changes throughout the year
    - **Bright/dark spots**: Indicate unusually hot or cold periods
    - **Gradient changes**: Show how rapidly temperatures change during different seasons
    """)

# Visualization is already displayed above

# This function has been replaced by render_temperature_gradient

# Function to handle temperature range slider changes is now implemented
# as a method in the SessionStateManager class

def get_temperature_styles() -> str:
    """
    Get CSS styles for temperature controls
    
    Returns:
        str: CSS styles for temperature controls
    """
    return """
    <style>
    /* Reduce bottom margin of slider container */
    .stSlider {
        margin-bottom: -15px !important;
    }
    
    /* Style for data range indicators */
    .data-range-indicator {
        position: relative;
        height: 25px;
        margin-top: 0px;
        margin-bottom: 10px;
        width: 100%;
    }
    .tick-mark {
        position: absolute;
        width: 2px;
        height: 10px;
        background-color: rgba(255,255,255,0.7);
        transform: translateX(-50%);
    }
    .tick-label {
        position: absolute;
        font-size: 10px;
        transform: translateX(-50%);
        top: 12px;
        white-space: nowrap;
        font-weight: bold;
    }
    
    /* Legend tick styles */
    .legend-tick-container {
        position: relative;
        height: 30px;
        margin-top: -15px;  /* Increased negative margin to move ticks closer to legend */
        width: 100%;
        max-width: 100%;
        overflow: visible;
        padding: 0 10px;
        box-sizing: border-box;
    }
    .legend-tick-mark {
        position: absolute;
        width: 2px;
        height: 8px;
        background-color: rgba(255,255,255,0.9);
        transform: translateX(-50%);
    }
    .legend-tick-label {
        position: absolute;
        font-size: 12px;
        color: rgba(255,255,255,0.9);
        transform: translateX(-50%);
        top: 5px;  /* Reduced from 8px to keep labels closer to tick marks */
        white-space: nowrap;
        font-weight: bold;
    }
    </style>
    """

def render_temperature_gradient(
    cmap_name: str,
    min_temp: float,
    max_temp: float,
    width: int = 600,
    height: int = 30
) -> plt.Figure:
    """
    Create a gradient bar visualization for temperature range selection
    
    Args:
        cmap_name: Name of the colormap to use
        min_temp: Minimum temperature value
        max_temp: Maximum temperature value
        width: Width of the gradient bar in pixels
        height: Height of the gradient bar in pixels
        
    Returns:
        Matplotlib figure with gradient bar
    """
    # Create figure with tight layout and no padding
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding
    
    # Create gradient
    gradient = np.linspace(0, 1, width)
    gradient = np.vstack((gradient, gradient))
    
    # Display gradient
    ax.imshow(gradient, aspect='auto', cmap=cmap_name)
    
    # Add temperature ticks
    num_ticks = 5  # Number of ticks to show
    tick_positions = np.linspace(0, width-1, num_ticks)
    tick_labels = np.linspace(min_temp, max_temp, num_ticks)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{t:.1f}°C" for t in tick_labels])
    
    # Style the ticks
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Make border transparent
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set tight layout with minimal padding
    plt.tight_layout(pad=0.1)
    
    return fig

def render_data_range_indicators(
    slider_min: float,
    slider_max: float,
    data_min: float,
    data_max: float
) -> None:
    """
    Render data range indicators below the temperature slider
    
    Args:
        slider_min: Minimum value of the slider
        slider_max: Maximum value of the slider
        data_min: Minimum temperature in the data
        data_max: Maximum temperature in the data
    """
    # Calculate positions for data min and max as percentages
    slider_range = slider_max - slider_min
    min_pos = ((data_min - slider_min) / slider_range) * 100
    max_pos = ((data_max - slider_min) / slider_range) * 100
    
    # Create HTML for the data range indicators
    html = f"""
    <div class="data-range-indicator">
        <div class="tick-mark" style="left: {min_pos}%;"></div>
        <div class="tick-label" style="left: {min_pos}%; font-size: 10px; color: rgba(100,200,255,0.9);">
            {data_min:.1f}°C<br><span style="font-size: 8px; opacity: 0.8;">data min</span>
        </div>
        <div class="tick-mark" style="left: {max_pos}%;"></div>
        <div class="tick-label" style="left: {max_pos}%; font-size: 10px; color: rgba(255,100,0,0.9);">
            {data_max:.1f}°C<br><span style="font-size: 8px; opacity: 0.8;">data max</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_temperature_range_slider(
    session: SessionStateManager,
    min_data_temp: float,
    max_data_temp: float
) -> Tuple[float, float]:
    """
    Render the temperature range slider with data range indicators
    
    Args:
        session: Session state manager
        min_data_temp: Minimum temperature in the data
        max_data_temp: Maximum temperature in the data
        
    Returns:
        Tuple[float, float]: Selected min and max temperature values
    """
    # Set slider range to use custom range limits
    slider_min = float(session.custom_range_min)
    slider_max = float(session.custom_range_max)
    
    # Calculate appropriate step size
    slider_step = (slider_max - slider_min) / 100  # 100 steps across the range
    
    # Store current values for validation
    current_min = float(session.min_temp)
    current_max = float(session.max_temp)
    
    # Create a slider with the current min/max values
    temp_range = st.slider(
        "Temperature Range",
        min_value=slider_min,
        max_value=slider_max,
        value=(current_min, current_max),
        step=slider_step,
        format="%.1f°C",
        key="temp_range_slider",
        help="Drag the handles to adjust the minimum and maximum temperature values for color mapping."
    )
    
    # Render data range indicators
    render_data_range_indicators(slider_min, slider_max, min_data_temp, max_data_temp)
    
    # Add reset button in a more prominent location with full width
    if st.button("Reset Temperature Range",
                key="reset_temp_range_button",
                help="Reset to data range for optimal visualization",
                use_container_width=True):
        # Set the temperature values to the data range
        session.update_temp_range(min_data_temp, max_data_temp, min_data_temp, max_data_temp)
        # Force rerun with the new values
        st.rerun()
    
    # Update session state if slider values changed
    if temp_range != (current_min, current_max):
        try:
            session.update_temp_range(temp_range[0], temp_range[1], min_data_temp, max_data_temp)
            # Force rerun to update the visualization
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    
    return temp_range

def render_temperature_inputs(
    session: SessionStateManager,
    temp_range: Tuple[float, float],
    min_data_temp: float,
    max_data_temp: float,
    slider_min: float,
    slider_max: float,
    slider_step: float
) -> None:
    """
    Render fine-tuning temperature input fields
    
    Args:
        session: Session state manager
        temp_range: Current temperature range (min, max)
        min_data_temp: Minimum temperature in the data
        max_data_temp: Maximum temperature in the data
        slider_min: Minimum value of the slider
        slider_max: Maximum value of the slider
        slider_step: Step size for the slider
    """
    st.subheader("Fine-tune Temperature Values")
    st.caption("Adjust temperature values with precise inputs:")
    
    temp_input_col1, temp_input_col2 = st.columns(2)
    
    with temp_input_col1:
        min_temp_input = st.number_input(
            "Min Temperature (°C)",
            value=temp_range[0],
            min_value=slider_min,
            max_value=temp_range[1] - slider_step,  # Ensure min < max
            step=slider_step,
            format="%.1f",
            help="Minimum temperature for color mapping."
        )
    
    with temp_input_col2:
        max_temp_input = st.number_input(
            "Max Temperature (°C)",
            value=temp_range[1],
            min_value=min_temp_input + slider_step,  # Ensure max > min
            max_value=slider_max,
            step=slider_step,
            format="%.1f",
            help="Maximum temperature for color mapping."
        )
    
    # Update session state based on number inputs
    if (min_temp_input, max_temp_input) != temp_range:
        try:
            session.update_temp_range(min_temp_input, max_temp_input, min_data_temp, max_data_temp)
            # Force rerun to update the visualization
            st.rerun()
        except ValueError as e:
            st.error(str(e))

def render_advanced_settings(
    session: SessionStateManager,
    min_data_temp: float,
    max_data_temp: float
) -> None:
    """
    Render advanced temperature settings
    
    Args:
        session: Session state manager
        min_data_temp: Minimum temperature in the data
        max_data_temp: Maximum temperature in the data
    """
    st.subheader("Advanced Settings")
    st.caption("Define the minimum and maximum selectable temperature values for the slider.")
    
    # Create two columns for min/max inputs
    custom_col1, custom_col2 = st.columns(2)
    
    with custom_col1:
        # Input for minimum range limit
        custom_min = st.number_input(
            "Minimum Selectable Temperature (°C)",
            value=float(session.custom_range_min),
            step=1.0,
            help="The lowest temperature value that can be selected on the slider."
        )
        session.custom_range_min = custom_min
    
    with custom_col2:
        # Input for maximum range limit
        custom_max = st.number_input(
            "Maximum Selectable Temperature (°C)",
            value=float(session.custom_range_max),
            step=1.0,
            help="The highest temperature value that can be selected on the slider."
        )
        session.custom_range_max = custom_max
    
    # Validate that min < max
    if custom_min >= custom_max:
        st.error("Minimum range limit must be less than maximum range limit.")
        # Reset to valid values
        custom_min = min(custom_min, custom_max - 1)
        custom_max = max(custom_max, custom_min + 1)
        session.custom_range_min = custom_min
        session.custom_range_max = custom_max
        st.rerun()
    
    # Reset button removed from advanced settings

def render_temperature_settings(
    session: SessionStateManager,
    min_data_temp: float,
    max_data_temp: float,
    df: pd.DataFrame
) -> None:
    """
    Render the temperature settings expander with all controls
    
    Args:
        session: Session state manager
        min_data_temp: Minimum temperature in the data
        max_data_temp: Maximum temperature in the data
        df: DataFrame containing temperature data
    """
    # Apply temperature styles
    st.markdown(get_temperature_styles(), unsafe_allow_html=True)
    
    # Temperature Range section
    st.subheader("Temperature Range")
    st.caption("""
    Adjust the temperature range to control how colors are mapped to temperatures in the visualization.
    
    - **Narrower range**: Creates higher contrast, making small temperature differences more visible
    - **Wider range**: Shows the full temperature spectrum with more subtle color transitions
    - **Reset button**: Returns to the optimal range for this dataset
    """)
    
    # Set slider range to use custom range limits
    slider_min = float(session.custom_range_min)
    slider_max = float(session.custom_range_max)
    slider_step = (slider_max - slider_min) / 100  # 100 steps across the range
    
    # Temperature information display removed as per user request
    
    # Render temperature range slider
    temp_range = render_temperature_range_slider(session, min_data_temp, max_data_temp)
    
    # No explanation needed as per user request
    
    # Render fine-tuning temperature inputs
    render_temperature_inputs(
        session,
        temp_range,
        min_data_temp,
        max_data_temp,
        slider_min,
        slider_max,
        slider_step
    )
    
    # Render advanced settings
    render_advanced_settings(session, min_data_temp, max_data_temp)

# Ticks are already displayed with the legend

# Input controls have been moved above the tabs

# Add temperature statistics in an expandable section
with st.expander("Temperature Data Details", expanded=False):
    st.markdown("### Temperature Extremes")
    st.markdown("This section shows the minimum and maximum temperatures recorded in the dataset.")
    
    stat_col1, stat_col2 = st.columns(2)

    # Find indices of maximum and minimum temperatures
    max_temp_index = df['temperature'].idxmax()
    min_temp_index = df['temperature'].idxmin()

    with stat_col1:
        st.markdown("##### Min. Temperature")
        min_time_str = df.iloc[min_temp_index]['time'].strftime('%Y-%m-%d %H:%M:%S')
        min_temp_str = f"{df.iloc[min_temp_index]['temperature']} °C"
        st.metric(label=min_time_str, value=min_temp_str, border=False)

    with stat_col2:
        st.markdown("##### Max. Temperature")
        max_time_str = df.iloc[max_temp_index]['time'].strftime('%Y-%m-%d %H:%M:%S')
        max_temp_str = f"{df.iloc[max_temp_index]['temperature']} °C"
        st.metric(label=max_time_str, value=max_temp_str, border=False)
    
    # Add some additional temperature statistics
    st.markdown("### Temperature Distribution")
    
    # Calculate temperature statistics
    avg_temp = df['temperature'].mean()
    median_temp = df['temperature'].median()
    std_temp = df['temperature'].std()
    
    # Display statistics
    st.markdown(f"""
    - **Average Temperature**: {avg_temp:.2f} °C
    - **Median Temperature**: {median_temp:.2f} °C
    - **Temperature Std. Deviation**: {std_temp:.2f} °C
    - **Temperature Range**: {min_data_temp:.2f} °C to {max_data_temp:.2f} °C
    """)

# Split visualization settings into multiple expandable sections in the sidebar

# 1. Colormap Selection
with st.sidebar.expander("Colormap Selection", expanded=False):
    # Colormap introduction removed as per user request
    
    colormap_config = colormap_selector(key_prefix="main_colormap", display_mode="grid", use_expander=False)
    
    # Update the style in session state
    if colormap_config['reverse']:
        session.style = f"{colormap_config['colormap']}_r"
    else:
        session.style = colormap_config['colormap']

# 2. Temperature Settings
with st.sidebar.expander("Temperature Settings", expanded=False):
    render_temperature_settings(session, min_data_temp, max_data_temp, df)

# Add location section in the sidebar
with st.sidebar.expander("Location", expanded=False):
    # Add location input field above the map
    sidebar_address = st.text_input(
        label="",
        value=address,
        key="sidebar_location",
        help="Enter an address (e.g., 'New York') or coordinates (e.g., '47.3769° N, 8.5417° E')"
    )
    
    # Update the map if the location changes
    if sidebar_address != address:
        st.rerun()
    
    # Create a DataFrame with the location coordinates for the map
    import pandas as pd
    map_data = pd.DataFrame({
        'lat': [session.location_coordinates.latitude],
        'lon': [session.location_coordinates.longitude]
    })
    
    # Display the map with the location
    st.map(data=map_data, zoom=10, use_container_width=True)


