import base64
import datetime
import io
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml
from github import GithubException
import os
import datetime
import pandas as pd
import requests

from github_file import repo, upload_file_to_github, REPO_NAME


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


import os
import datetime
import pandas as pd
import requests


def initialize_summary_file(network, code, deployed=True):
    """
    在指定的目录中创建一个包含 headers 的空 CSV 文件。如果文件已存在，则不会重新创建。

    Args:
    - network (str): 网络代码。
    - code (str): 站点代码。
    - deployed (bool): 如果为 True，则在 GitHub 上创建文件；否则在本地创建。

    Returns:
    - str: 创建的文件路径（如果在本地创建）或 GitHub 文件路径（如果在 GitHub 上创建）。
    """
    repo_name = st.secrets["REPO_NAME"]  # 从配置中获取 GitHub 仓库名称
    repo_dir = os.path.join("data", f"{network}.{code}").replace("\\", "/")
    headers = ["network", "code", "date", "unique_id", "provider", "event_id", "time", "lat", "long", "mag",
               "mag_type", "depth", "epi_distance", "p_predicted", "s_predicted", "p_detected", "s_detected",
               "p_confidence", "s_confidence", "p_error", "s_error", "catalogued", "detected", "plot_path"]

    if deployed:
        # 在 GitHub 上的文件名改为固定的名称
        summary_file_url = f"https://raw.githubusercontent.com/{repo_name}/main/{repo_dir}/processed_earthquakes_summary.csv"

        # 检查文件是否已存在
        try:
            response = requests.head(summary_file_url)
            if response.status_code == 200:
                print(f"File '{summary_file_url}' already exists on GitHub. No new file created.")
                return summary_file_url
            else:
                print(f"File not found on GitHub. Proceeding to create a new one. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error checking file on GitHub: {str(e)}. Proceeding to create a new one.")

        # 创建一个空的 DataFrame，并设置 columns
        empty_data = pd.DataFrame(columns=headers)

        # 将空的 DataFrame 保存为 CSV 文件并上传到 GitHub
        temp_file_path = "temp_empty_summary.csv"
        empty_data.to_csv(temp_file_path, index=False)
        upload_file_to_github(temp_file_path, f"{repo_dir}/processed_earthquakes_summary.csv")
        os.remove(temp_file_path)  # 删除临时文件
        print(f"Empty summary file '{summary_file_url}' uploaded to GitHub.")

        return summary_file_url

    else:
        # 本地路径
        station_folder = os.path.join(os.getcwd(), "data", f"{network}.{code}")
        total_file_path = os.path.join(station_folder, "processed_earthquakes_summary.csv")

        # 检查文件是否已存在
        if os.path.exists(total_file_path):
            print(f"File '{total_file_path}' already exists locally. No new file created.")
            return total_file_path

        # 创建文件夹（如果不存在）
        if not os.path.exists(station_folder):
            os.makedirs(station_folder, exist_ok=True)

        # 创建一个空的 DataFrame，并设置 columns
        empty_data = pd.DataFrame(columns=headers)

        # 保存 CSV 文件
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
        repo_name = st.secrets["REPO_NAME"]

        # 构建raw链接
        summary_file_url = f"https://raw.githubusercontent.com/{repo_name}/main/{station_folder}/processed_earthquakes_summary.csv"
        print("Reading summary file from GitHub")

        try:
            # 发送GET请求获取CSV内容
            response = requests.get(summary_file_url)
            response.raise_for_status()  # 如果请求失败，会抛出异常

            # 检查文件内容是否为空
            if response.text.strip() == "":
                print("File is empty.")
                return None, "file_empty"

            # 使用io.StringIO读取CSV数据
            df = pd.read_csv(io.StringIO(response.text))
            print("Summary file loaded successfully.")
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

        file_path = os.path.join(station_folder_path, "processed_earthquakes_summary.csv")
        try:
            df = pd.read_csv(file_path)
            print("File loaded from local filesystem.")
            return df, "loaded"
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, "not_found"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, "error_reading"


def load_summary_to_session():
    # 尝试加载事件汇总 CSV
    df, status = read_summary_csv(
        network=st.session_state.network,
        code=st.session_state.station_code
    )

    # 如果 DataFrame 是 None，表示文件未找到或加载失败
    if status == "not_found":
        print("Total events summary file not found. No summary file will be created.")
        return "not_exist"

    elif status == "error_reading":
        print("Error occurred while reading the summary file. Please check the file manually.")
        return "error"

    if df.empty:
        print("Total events summary file is empty.")
        st.session_state.df = df
        return "exist_empty"
    else:
        st.session_state.df = df
        return "exist_loaded"


# Function to load settings from a YAML file


def load_config_to_df(filename='user_config.yaml', deployed=True):
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


def save_config_to_yaml(session_state, filename='user_config.yaml', deployed=True):
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


def filtering_options_and_statistics(df, enable_date_filtering=True):
    """
    Render filtering options and statistics inside an expander.

    Args:
    - df (pd.DataFrame): The DataFrame containing the earthquake data.
    - enable_date_filtering (bool): Whether to enable date filtering options.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    with st.expander("Filtering Options and Statistics", expanded=False):
        st.subheader("Filtering Options")

        # Define confidence threshold slider
        confidence_threshold = st.slider(
            'Confidence Threshold',
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            key='confidence_slider'
        )

        # Define p_error slider
        p_error_threshold = st.slider(
            'P Error Threshold (seconds)',
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            key='p_error_slider'
        )

        # Checkbox to filter by mag_type Mw
        filter_mw_only = st.toggle("Only show earthquakes of scale 'Mw'", value=False)

        # Create a copy of the DataFrame and apply filters
        df_filtered = df.copy()
        df_filtered['p_confidence'] = pd.to_numeric(df_filtered['p_confidence'], errors='coerce')

        # Convert 'date' column to datetime format if it is not already
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce').dt.date

        # Filter based on p_confidence threshold
        df_filtered = df_filtered.loc[df_filtered['p_confidence'] >= confidence_threshold]

        # Apply Mw filter if checkbox is selected
        if filter_mw_only:
            df_filtered = df_filtered[df_filtered['mag_type'].str.lower() == 'mw']

        # Ensure 'detected' and 'catalogued' are boolean
        df_filtered['detected'] = df_filtered['detected'].astype(bool)
        df_filtered['catalogued'] = df_filtered['catalogued'].astype(bool)

        # Convert 'p_error' to numeric
        df_filtered['p_error'] = pd.to_numeric(df_filtered['p_error'], errors='coerce')

        # Apply p_error filter
        df_filtered = df_filtered.loc[df_filtered['p_error'].abs() <= p_error_threshold]
        st.divider()

        # 新的 Statistics 容器，显示过滤前后的地震数量
        st.subheader("Filtering Result Statistics")

        # 使用列布局
        col1, col2 = st.columns(2)

        with col1:
            # 过滤前的地震数量，确保 catalogued 和 detected 都为 True
            total_earthquakes_before_filter = df[
                (df['catalogued'] == True) & (df['detected'] == True)
                ].shape[0]

            # 显示过滤前的统计信息
            st.metric(label="Number Of Earthquakes In Summary", value=total_earthquakes_before_filter)

        with col2:
            # 过滤后的地震数量
            total_earthquakes_after_filter = df_filtered[
                (df_filtered['catalogued'] == True) & (df_filtered['detected'] == True)
                ].shape[0]

            # 显示过滤后的统计信息
            st.metric(label="Number Of Earthquakes After Filtering", value=total_earthquakes_after_filter)

        # 添加选择日期过滤的选项（仅当 enable_date_filtering 为 True 时显示）
        if enable_date_filtering:
            st.divider()
            date_filter_option = st.radio(
                "See Filtered Earthquakes By Date",
                options=["Show all dates", "Select a specific date"],
                index=0
            )

            if date_filter_option == "Select a specific date":
                # 从过滤后的 df_filtered 中获取唯一日期
                available_dates = sorted(df_filtered['date'].dropna().unique())

                # 创建一个日期选择器，用户可以选择要过滤的日期
                selected_date = st.selectbox(
                    "Select a date to filter",
                    available_dates,
                    format_func=lambda x: x.strftime('%Y-%m-%d')
                )

                # 根据选择的日期再次过滤数据框
                df_filtered = df_filtered[df_filtered['date'] == selected_date]

    return df_filtered


def sidebar_navigation():
    """
    Render the sidebar navigation with links to different pages.
    """
    st.sidebar.title("Navigation")
    st.sidebar.page_link('Home.py', label='Home')
    st.sidebar.page_link('pages/Daily_Report_page.py', label='Daily Report')
    st.sidebar.page_link('pages/Event_Report_page.py', label='Event Report')
    st.sidebar.page_link('pages/Statistics_page.py', label='Statistics')
    st.sidebar.page_link('pages/Settings_page.py', label='Settings')
