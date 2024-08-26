import base64
import datetime
import json
import os
import re
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml
from github import GithubException
import yaml
import os
import base64
import io

from github_file import repo, upload_file_to_github


def check_file_exists_in_github(repo_file_path):
    """
    检查 GitHub 仓库中的文件是否存在。

    Args:
    - repo_file_path (str): 仓库中的文件路径。

    Returns:
    - bool: 如果文件存在返回 True，否则返回 False。
    """
    try:
        repo.get_contents(repo_file_path)
        return True
    except GithubException as e:
        if e.status == 404:
            return False
        else:
            raise


def create_empty_summary_file(network, code, deployed=True):
    """
    在指定的目录中创建一个包含 headers 的空 CSV 文件。

    Args:
    - network (str): 网络代码。
    - code (str): 站点代码。
    - deployed (bool): 如果为 True，则在 GitHub 上创建文件；否则在本地创建。

    Returns:
    - str: 创建的文件路径（如果在本地创建）或 GitHub 文件路径（如果在 GitHub 上创建）。
    """
    repo_dir = os.path.join("data", f"{network}.{code}")
    headers = ["network", "code", "date", "unique_id", "provider", "event_id", "time", "lat", "long", "mag",
               "mag_type", "depth", "epi_distance", "p_predicted", "s_predicted", "p_detected", "s_detected",
               "p_confidence", "s_confidence", "p_error", "s_error", "catalogued", "detected", "plot_path"]

    # 创建一个空的 DataFrame，并设置 columns
    empty_data = pd.DataFrame(columns=headers)

    if deployed:
        # 在 GitHub 上的文件名改为固定的名称
        summary_file_path = f"{repo_dir}/processed_earthquakes_summary.csv"
        repo_dir = repo_dir.replace("\\", "/")

        # 将空的 DataFrame 保存为 CSV 文件并上传到 GitHub
        temp_file_path = "temp_empty_summary.csv"
        empty_data.to_csv(temp_file_path, index=False)
        upload_file_to_github(temp_file_path, summary_file_path)
        print(f"Empty summary file '{summary_file_path}' uploaded to GitHub.")

        return summary_file_path

    else:
        # 本地路径添加时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        station_folder = os.path.join(os.getcwd(), "data", f"{network}.{code}")
        if not os.path.exists(station_folder):
            os.makedirs(station_folder, exist_ok=True)

        total_file_path = os.path.join(station_folder, f"processed_earthquakes_summary.csv")
        empty_data.to_csv(total_file_path, index=False)
        print(f"Empty summary file created locally at '{total_file_path}'.")

        return total_file_path

def read_summary_csv(network, code, deployed=True):
    """
    Static method to read the processed earthquakes summary CSV file.

    Args:
    - network (str): The network code.
    - code (str): The station code.
    - deployed (bool): If True, read from GitHub; otherwise, read from the local filesystem.

    Returns:
    - pd.DataFrame: DataFrame containing the processed earthquake summary data.
    - str: Status message indicating the result of the operation.
    """

    station_folder = os.path.join("data", f"{network}.{code}").replace("\\", "/")

    if deployed:
        # 构建raw链接
        summary_file_url = f"https://raw.githubusercontent.com/QingyuanZhangTony/test/main/{station_folder}/processed_earthquakes_summary.csv"

        try:
            # 发送GET请求获取CSV内容
            response = requests.get(summary_file_url)
            response.raise_for_status()  # 如果请求失败，会抛出异常

            # 检查文件内容是否为空
            if response.text.strip() == "":
                return None, "file_empty"

            # 使用io.StringIO读取CSV数据
            df = pd.read_csv(io.StringIO(response.text))
            return df, "loaded"
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred while accessing {summary_file_url}: {str(http_err)}")
            return None, "not_found"
        except Exception as e:
            print(f"An error occurred while reading the file from {summary_file_url}: {str(e)}")
            return None, "error_reading"

    else:
        # 从本地文件系统读取文件
        base_dir = os.getcwd()
        station_folder_path = os.path.join(base_dir, station_folder)

        try:
            file_path = os.path.join(station_folder_path, "processed_earthquakes_summary.csv")
            df = pd.read_csv(file_path)
            return df, "loaded"
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, "not_found"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, "error_reading"

    return None, "not_found"

def load_summary_to_session():
    # 尝试加载事件汇总 CSV
    df, status = read_summary_csv(
        network=st.session_state.network,
        code=st.session_state.station_code
    )

    # 如果 DataFrame 是 None，表示文件未找到或加载失败
    if status == "not_found":
        st.warning("Total events summary file not found. Creating an empty summary file...")

        # 尝试创建一个空的 summary 文件，只有在文件完全不存在时才创建
        if not check_file_exists_in_github(
                os.path.join("data", f"{st.session_state.network}.{st.session_state.station_code}")):
            success = create_empty_summary_file(
                network=st.session_state.network,
                code=st.session_state.station_code,
                deployed=True  # 如果你是在部署模式下运行，请确保这里设置为 True
            )

            if success:
                st.success("Empty summary file created successfully.")
                df, status = read_summary_csv(
                    network=st.session_state.network,
                    code=st.session_state.station_code
                )

                if status != "loaded":
                    st.error("Failed to load the newly created empty summary file.")
                    return "not_exist"
                else:
                    st.session_state.df = df
                    return "exist_empty"
            else:
                st.error("Failed to create an empty summary file.")
                return "not_exist"
        else:
            st.error("Summary file exists but could not be loaded.")
            return "not_exist"

    elif status == "error_reading":
        st.error("Error occurred while reading the summary file. Please check the file manually.")
        return "error"

    if df.empty:
        st.warning("Total events summary file is empty.")
        st.session_state.df = df  #
        return "exist_empty"
    else:
        st.session_state.df = df
        return "exist_loaded"


# Function to load settings from a YAML file


def load_config_to_df(filename='default_config.yaml', deployed=True):
    """
    Load the configuration from a YAML file.

    Args:
    - filename (str): The name of the YAML file.
    - deployed (bool): If True, load the file from GitHub; otherwise, load it from the local filesystem.

    Returns:
    - dict: A dictionary containing the configuration.
    """

    if deployed:
        # GitHub mode: Load the file from the GitHub repository
        try:
            contents = repo.get_contents(filename)
            file_content = base64.b64decode(contents.content).decode("utf-8")
            config = yaml.safe_load(io.StringIO(file_content))
        except GithubException as e:
            print(f"An error occurred while accessing GitHub path '{filename}': {str(e)}")
            return None
    else:
        # Local mode: Load the file from the local filesystem
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None
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


def save_config_to_yaml(session_state, filename='default_config.yaml', deployed=True):
    """
    Save the configuration from the session state to a YAML file.

    Args:
    - session_state (dict): The session state containing the configuration.
    - filename (str): The name of the YAML file.
    - deployed (bool): If True, save the file to GitHub; otherwise, save it to the local filesystem.
    """

    if deployed:
        # GitHub mode: Load the existing configuration from GitHub
        try:
            contents = repo.get_contents(filename)
            existing_config_content = base64.b64decode(contents.content).decode("utf-8")
            existing_config = yaml.safe_load(io.StringIO(existing_config_content))
        except GithubException as e:
            print(f"An error occurred while accessing GitHub path '{filename}': {str(e)}")
            return False
    else:
        # Local mode: Load the existing configuration from the local filesystem
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False
        with open(filename, 'r') as file:
            existing_config = yaml.safe_load(file)

    # Create a new config dictionary that only includes keys present in the existing config
    config = {key: session_state[key] for key in existing_config if key in session_state}

    if deployed:
        # Convert the config dictionary to YAML format
        config_yaml = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)

        # Upload the updated configuration back to GitHub
        try:
            repo.update_file(contents.path, "Update configuration file", config_yaml, contents.sha)
            print(f"Configuration saved to GitHub: {filename}")
        except GithubException as e:
            print(f"An error occurred while saving the configuration to GitHub: {str(e)}")
            return False
    else:
        # Save the filtered configuration dictionary to the local YAML file
        with open(filename, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved locally: {filename}")

    return True


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
