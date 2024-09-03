import datetime

import pandas as pd
import streamlit as st

from logic import initialisation_logic
from streamlit_utils import load_config_to_session, sidebar_navigation, \
    load_summary_to_session

# Sidebar navigation
sidebar_navigation()

# Load default settings
load_config_to_session()

# Check if it's the first time running
if not st.session_state.get('initialized', False):
    st.title("Welcome to Earthquake Monitoring And Report")
    st.header("First-Time Setup Required")

    st.write(
        "It looks like this is your first time running the application. Please provide the following information to get started.")

    # User input for configuration parameters
    network = st.text_input("Network Name", st.session_state.get('network', ''))
    station_code = st.text_input("Station Code", st.session_state.get('station_code', ''))

    # Data Provider URL input with option to select Raspberry Shake or custom URL
    data_provider_choice = st.radio(
        "Select Data Provider",
        options=["I am using a RaspberryShake", "Enter Custom URL"],
        index=0  # Default to Raspberry Shake option
    )

    if data_provider_choice == "I am using a RaspberryShake":
        data_provider_url = "https://data.raspberryshake.org"
    else:
        # Display an input box for custom URL without a label
        data_provider_url = st.text_input("")

    # User input for email address to receive reports
    email_recipient = st.text_input("Email Address for Receiving Reports", '')

    # Save settings button
    if st.button("Save Settings and Start"):
        initialisation_logic(network, station_code, data_provider_url, email_recipient)

else:
    # If already initialized, show main content
    st.title("Earthquake Monitoring And Report")
    st.header(f"Welcome back, {st.session_state['station_code']}")

    st.write("You're all set up and ready to start monitoring earthquakes.")
    st.divider()

    with st.container():
        st.write(
            "This program was designed and implemented as part of my dissertation project for the MDS program at Durham University. "
            "Please refer to the user manual for guidance on getting started. "
            "If you have any questions or need assistance, feel free to contact me at xmpg69@durham.ac.uk."
        )

    summary_status = load_summary_to_session()

    st.divider()
    # 检查 summary 状态
    if summary_status == "exist_loaded":
        with st.container():
            st.subheader("Detection Update")

            # 获取今天日期
            today = datetime.date.today()

            # 将 'date' 列转换为 datetime 对象
            st.session_state.df['date'] = pd.to_datetime(st.session_state.df['date']).dt.date

            # 计算总匹配成功的事件数
            total_matched_events = \
                st.session_state.df[st.session_state.df['catalogued'] & st.session_state.df['detected']].shape[0]

            # 计算昨天的匹配成功事件数
            yesterday = today - datetime.timedelta(days=1)
            yesterday_matched_events = st.session_state.df[
                (st.session_state.df['catalogued'] & st.session_state.df['detected']) & (
                        st.session_state.df['date'] == yesterday)
                ].shape[0]

            # 计算前天的匹配成功事件数（用于计算delta）
            day_before_yesterday = today - datetime.timedelta(days=2)
            day_before_yesterday_matched_events = st.session_state.df[
                (st.session_state.df['catalogued'] & st.session_state.df['detected']) & (
                        st.session_state.df['date'] == day_before_yesterday)
                ].shape[0]

            # 计算过去30天的匹配成功事件数
            last_30_days = today - datetime.timedelta(days=30)
            last_30_days_matched_events = st.session_state.df[
                (st.session_state.df['catalogued'] & st.session_state.df['detected']) & (
                        st.session_state.df['date'] >= last_30_days)
                ].shape[0]

            # 计算前30天的匹配成功事件数（用于计算delta）
            previous_30_days = today - datetime.timedelta(days=60)
            previous_30_days_matched_events = st.session_state.df[
                (st.session_state.df['catalogued'] & st.session_state.df['detected']) &
                (st.session_state.df['date'] >= previous_30_days) &
                (st.session_state.df['date'] < last_30_days)
                ].shape[0]

            # 计算delta
            yesterday_delta = yesterday_matched_events - day_before_yesterday_matched_events
            last_30_days_delta = last_30_days_matched_events - previous_30_days_matched_events

            # 展示在页面中
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Matched Events", total_matched_events)
            col2.metric("Matched Events Yesterday", yesterday_matched_events, delta=yesterday_delta)
            col3.metric("Matched Events Last 30 Days", last_30_days_matched_events, delta=last_30_days_delta)

    elif summary_status == "exist_empty":
        st.warning("Total events summary file is empty. No metrics to display.")
    else:
        st.error("Summary file could not be loaded. Metrics are unavailable.")
