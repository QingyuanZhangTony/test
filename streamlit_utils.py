import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import yaml


def read_summary_csv(network, code):
    """
    Static method to read the processed earthquakes summary CSV file.

    Args:
    - network (str): The network code.
    - code (str): The station code.

    Returns:
    - pd.DataFrame: DataFrame containing the processed earthquake summary data.
    """
    # Construct the path to the 'processed_earthquakes_summary.csv' file
    base_dir = os.getcwd()
    station_folder = os.path.join(base_dir, "data", f"{network}.{code}")
    file_path = os.path.join(station_folder, 'processed_earthquakes_summary.csv')

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def load_summary_to_session():
    # 尝试加载事件汇总 CSV
    df = read_summary_csv(
        network=st.session_state.network,
        code=st.session_state.station_code
    )

    # 如果 DataFrame 是 None，表示文件未找到或加载失败
    if df is None:
        st.warning("Total events summary file not found or could not be loaded.")
        return "not_exist"
    elif df.empty:
        st.warning("Total events summary file is empty.")
        return "exist_empty"
    else:
        st.session_state.df = df
        return "exist_loaded"




# Function to load settings from a YAML file
def load_config_to_df(filename='default_config.yaml'):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    # Convert list to comma-separated string for specific fields if necessary
    if 'catalog_providers' in config and isinstance(config['catalog_providers'], list):
        config['catalog_providers'] = ', '.join(config['catalog_providers'])
    return config


def load_config_to_session():
    # Load default settings
    config_file = load_config_to_df()

    # Initialize session state variables
    for key, value in config_file.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_config_to_yaml(session_state, filename='default_config.yaml'):
    # Load the existing configuration to determine which keys should be saved
    with open(filename, 'r') as file:
        existing_config = yaml.safe_load(file)

    # Create a new config dictionary that only includes keys present in the existing config
    config = {key: session_state[key] for key in existing_config if key in session_state}

    with open(filename, 'w') as file:
        # Save the filtered configuration dictionary to the YAML file
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)


def initialize_state_defaults():
    state_defaults = {
        # Station Data
        'station_downloaded': False,
        'is_downloading': False,

        # Catalog Data
        'catalog_downloaded': False,
        'is_catalog_downloading': False,
        'event_count': 0,
        'catalog_provider': "Not Downloaded",
        'selected_projection': "local",

        # Process Stream Data
        'stream_processed': False,
        'is_processing': False,

        # Detect Phases
        'phases_detected': False,
        'is_detecting': False,
        'p_count': 0,
        's_count': 0,

        # Match Events
        'matching_completed': False,
        'is_matching': False,
        'matching_summary': "",

        # Generate Report
        'report_generated': False,
        'is_generating_report': False,

        # Send Email
        'email_sent': False,
        'is_sending_email': False
    }

    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_matched_earthquakes(df, simplified, p_only, date_str):
    st.subheader("Catalog Plot of All Earthquakes")

    # 过滤出符合条件的地震事件
    matched_earthquakes = df[(df['detected'] == True) & (df['catalogued'] == True) & (df['date'] == date_str)]

    if not matched_earthquakes.empty:
        st.subheader("Matched Earthquakes Details")

        if len(matched_earthquakes) > 5:
            options = matched_earthquakes['unique_id'].tolist()
            selected_eq_id = st.selectbox("Select Earthquake", options)
            selected_earthquake = matched_earthquakes[matched_earthquakes['unique_id'] == selected_eq_id].iloc[0]
            earthquakes_to_display = [(None, selected_earthquake)]
        else:
            tabs = st.tabs(matched_earthquakes['unique_id'].tolist())
            earthquakes_to_display = zip(tabs, matched_earthquakes.iterrows())

        for tab, earthquake in earthquakes_to_display:
            container = tab if tab else st.container()
            with container:
                if earthquake['plot_path']:
                    st.image(earthquake['plot_path'])

                # First line: Time, Location, Magnitude
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Time:** {pd.to_datetime(earthquake['time']).strftime('%Y-%m-%d %H:%M:%S')}")
                with col2:
                    st.write(f"**Location:** {float(earthquake['lat']):.2f}, {float(earthquake['long']):.2f}")
                with col3:
                    st.write(f"**Magnitude:** {float(earthquake['mag']):.2f} {earthquake['mag_type']}")

                if not simplified:
                    event_id_display = earthquake['event_id'].split(':', 1)[-1] if ':' in earthquake['event_id'] else \
                        earthquake['event_id']
                    st.write(f"**Event ID:** {event_id_display}")

                # Second line: Distance, Depth, Unique ID
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.write(f"**Epicentral Distance:** {float(earthquake['epi_distance']):.2f} km")
                with col5:
                    st.write(f"**Depth:** {float(earthquake['depth']):.2f} km")
                with col6:
                    st.write(f"**Unique ID:** {earthquake['unique_id']}")

                # Third line: P Predicted, P Detected, P Error
                col7, col8, col9 = st.columns(3)
                with col7:
                    st.write(
                        f"**P Predicted:** {pd.to_datetime(earthquake['p_predicted']).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(earthquake['p_predicted']) else 'N/A'}")
                with col8:
                    st.write(
                        f"**P Detected:** {pd.to_datetime(earthquake['p_detected']).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(earthquake['p_detected']) else 'N/A'}")
                with col9:
                    st.write(f"**P Error:** {earthquake['p_error']}")

                # Fourth line: P Confidence (if applicable)
                col10, col11, col12 = st.columns(3)
                with col10:
                    st.write(f"**P Confidence:** {earthquake['p_confidence']}")

                # Add more lines or data as needed based on the DataFrame's content


# Function to create evenly distributed ticks for different Color-by options
def create_ticks(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    tick_vals = np.linspace(min_val, max_val, 10)
    tick_texts = [f"{val:.2f}" for val in tick_vals]
    return tick_vals, tick_texts


def load_plate_boundaries(file_name='PB2002_boundaries.json'):
    """
    Load the plate boundaries from a GeoJSON file.

    Args:
        file_name (str): The name of the GeoJSON file to load.

    Returns:
        tuple: A tuple containing two lists, lines_lat and lines_lon, which
               represent the latitude and longitude coordinates of the plate boundaries.
    """
    # Define the path to the local JSON file
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, 'asset', file_name)

    # Read the JSON data from the local file
    with open(file_path, 'r') as file:
        plate_boundaries = json.load(file)

    # Extract coordinates from the GeoJSON for each line segment
    lines_lat = []
    lines_lon = []

    for feature in plate_boundaries['features']:
        coordinates = feature['geometry']['coordinates']
        if feature['geometry']['type'] == "LineString":
            lon, lat = zip(*coordinates)
            lines_lat.append(lat)
            lines_lon.append(lon)
        elif feature['geometry']['type'] == "MultiLineString":
            for line in coordinates:
                lon, lat = zip(*line)
                lines_lat.append(lat)
                lines_lon.append(lon)

    return lines_lat, lines_lon


def update_status(progress, message, update_progress=None):
    """
    通用的状态更新函数。用于在命令行输出和UI更新之间切换。

    Parameters:
    -----------
    progress : float
        进度值。
    message : str
        要显示的消息。
    update_progress : function, optional
        用于更新UI进度条的回调函数。
    """
    print(message)  # 命令行输出步骤
    if update_progress:
        update_progress(progress, message)  # 更新UI的进度条

