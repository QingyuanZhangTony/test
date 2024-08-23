import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
from logic import render_interactive_map, render_density_mapbox
from streamlit_utils import load_config_to_session, load_summary_to_session, save_config_to_yaml

# 加载配置到 session
load_config_to_session()

# 模拟启动时保存设置
if 'settings_initialized' not in st.session_state:
    st.session_state.settings_initialized = True
    save_config_to_yaml(st.session_state)

# 读取汇总数据到 session
summary_status = load_summary_to_session()

# Page title
st.title(f"Deployment Statistics for {st.session_state.station_code}")
st.divider()

# 检查 summary 状态
if summary_status == "exist_loaded":

    # Container for Filtering Options and DataFrame Filtering
    with st.container():
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
        df_filtered = st.session_state.df.copy()
        df_filtered['p_confidence'] = pd.to_numeric(df_filtered['p_confidence'], errors='coerce')

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

    # Container for Total No. Detect Over Station Life Time
    with st.container():
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce').dt.date

        # Ensure data is sorted by date
        df_filtered = df_filtered.sort_values(by='date')

        # Total number of detected events over station lifetime (cumulative count)
        df_filtered['cumulative_detected'] = df_filtered['detected'].cumsum()

        # Ensure cumulative count is non-decreasing
        df_filtered['cumulative_detected'] = df_filtered['cumulative_detected'].cummax()

        cumulative_detected = df_filtered.groupby('date')['cumulative_detected'].last().reset_index()
        cumulative_detected_chart = alt.Chart(cumulative_detected).mark_line().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('cumulative_detected:Q', title='Cumulative No. Detect'),
            tooltip=['date:T', 'cumulative_detected:Q']
        ).properties(
            height=300,
            width=700,
            title='Cumulative No. Detect Over Station Life Time'
        )

        st.altair_chart(cumulative_detected_chart, use_container_width=True)

    # Scatter Plot for Detected Earthquakes Only
    with st.container():
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce').dt.date

        # Filter for detected and catalogued events
        df_detected = df_filtered[(df_filtered['catalogued'] == True) & (df_filtered['detected'] == True)].copy()

        # Calculate the maximum magnitude and adjust the Y-axis domain
        max_magnitude = df_detected['mag'].max()
        y_max = (max_magnitude // 1) + 1  # Round up to the nearest whole number

        base_chart = alt.Chart(df_detected).mark_point(
            filled=True,
            opacity=1,
            size=50
        ).encode(
            x=alt.X('epi_distance:Q', title=f'Epicentral Distance To Station {st.session_state.station_code} (km)'),
            y=alt.Y('mag:Q', title='Magnitude', scale=alt.Scale(zero=False, domain=[0, y_max]),
                    axis=alt.Axis(tickCount=int(y_max + 1))),
            color=alt.Color('p_confidence:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(
                title='Confidence  ', titleOrient='left', orient='right', titleLimit=0, gradientLength=300
            )),
            order=alt.Order(
                'detected',  # Ensures detected events are drawn on top of not detected ones
                sort='ascending'
            ),
            tooltip=[
                alt.Tooltip('time:T', title='Date and Time', format='%Y-%m-%d %H:%M:%S'),
                alt.Tooltip('mag:Q', title='Magnitude'),
                alt.Tooltip('mag_type:N', title='Magnitude Type'),
                alt.Tooltip('lat:Q', title='Latitude'),
                alt.Tooltip('long:Q', title='Longitude'),
                alt.Tooltip('unique_id:N', title='Unique ID'),
                alt.Tooltip('epi_distance:Q', title='Epicentral Distance (km)'),
                alt.Tooltip('depth:Q', title='Depth (km)'),
                alt.Tooltip('p_confidence:Q', title='P-Wave Confidence'),
                alt.Tooltip('p_error:N', title='P-Wave Error'),
                alt.Tooltip('s_confidence:N', title='S-Wave Confidence'),
                alt.Tooltip('s_error:N', title='S-Wave Error')
            ]
        ).properties(
            height=500,
            title='Catalogued Earthquake Detection Overview'
        )

        st.altair_chart(base_chart, use_container_width=True)

    # Interactive Map Container
    with st.container():
        render_interactive_map(
            df_filtered,
            station_info={
                "latitude": st.session_state.station_latitude,
                "longitude": st.session_state.station_longitude,
                "code": st.session_state.station_code
            },
            title="Detected Catalogued Earthquakes Map Plot"
        )

    # Density Mapbox Container
    with st.container():
        # Display density mapbox using the provided function
        render_density_mapbox(
            df_filtered,
            station_info={
                "latitude": st.session_state.station_latitude,
                "longitude": st.session_state.station_longitude,
                "code": st.session_state.station_code
            },
            title="Density Map of Detected Earthquakes"
        )

elif summary_status == "exist_empty":
    st.warning("The summary file is empty. Please check the source or generate new data.")
else:  # summary_status == "not_exist"
    st.warning("The summary file does not exist. Please generate a report first.")
