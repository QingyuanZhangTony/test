import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
from logic import render_interactive_map, render_density_mapbox
from streamlit_utils import load_config_to_session, load_summary_to_session, save_config_to_yaml, \
    filtering_options_and_statistics, sidebar_navigation

# 加载配置到 session
load_config_to_session()
# Sidebar navigation
sidebar_navigation()
# 模拟启动时保存设置
if 'settings_initialized' not in st.session_state:
    st.session_state.settings_initialized = True
    save_config_to_yaml(st.session_state)

# 读取汇总数据到 session
summary_status = load_summary_to_session()


# Page title
st.title(f"Event Report")
st.divider()

# 检查 summary 状态
if summary_status == "exist_loaded":

    df_filtered = filtering_options_and_statistics(st.session_state.df.copy())

    # Interactive Map Container
    with st.container():
        render_interactive_map(
            df_filtered,
            station_info={
                "latitude": st.session_state.station_latitude,
                "longitude": st.session_state.station_longitude,
                "code": st.session_state.station_code
            },
            title="Select An Earthquake On The Map To Display Report",
        )

elif summary_status == "exist_empty":
    st.warning("The summary file is empty. Please check the source or generate new data.")
else:  # summary_status == "not_exist"
    st.warning("The summary file does not exist. Please generate a report first.")
