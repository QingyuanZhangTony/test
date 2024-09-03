import datetime
import os

import time
from urllib.error import URLError

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException

from catalog import CatalogData
from report import Report, DailyReport, EventReport
from station import Station
from streamlit_utils import load_plate_boundaries, create_ticks, update_status, load_config_to_df, read_summary_csv, \
    save_config_to_yaml, initialize_summary_file


def download_catalogue_logic(network, station_code, data_provider_url, report_date, latitude, longitude,
                             radmin, radmax, minmag, maxmag, catalogue_providers, update_status_func=None):
    # 创建Station对象用于catalog逻辑
    station = Station(network, station_code, data_provider_url, report_date, latitude=float(latitude),
                      longitude=float(longitude))
    catalog = CatalogData(station, radmin=radmin, radmax=radmax, minmag=minmag, maxmag=maxmag,
                          catalogue_providers=catalogue_providers)

    # 输出开始下载的日志
    update_status(0, f"Starting catalog download for {report_date.strftime('%Y-%m-%d')}...", update_status_func)

    attempts = 0
    progress_increment = 5  # Each provider attempt adds 5% to the progress
    while attempts < 2:
        for i, provider in enumerate(catalogue_providers):
            try:
                update_status(
                    progress_increment * i + (attempts * len(catalogue_providers) * progress_increment),
                    f"Trying provider {provider}...", update_status_func)

                result = catalog.request_catalogue_from_provider(provider)

                if result:
                    update_status(100, f"Catalog downloaded from {provider} with {len(result.events)} events!",
                                  update_status_func)
                    update_status(100, "Catalog Downloaded.", update_status_func)
                    print("-" * 50)
                    return station, catalog, f"Catalog downloaded from {provider}. Number of events: {len(result.events)}."

            except FDSNNoDataException:
                next_provider = catalogue_providers[i + 1] if i + 1 < len(catalogue_providers) else 'no more providers'
                update_status(0, f"No data available from {provider}. Trying {next_provider}", update_status_func)
            except FDSNException:
                next_provider = catalogue_providers[i + 1] if i + 1 < len(catalogue_providers) else 'no more providers'
                update_status(0, f"Error fetching earthquake data from {provider}. Trying {next_provider}",
                              update_status_func)
            except Exception as e:
                next_provider = catalogue_providers[i + 1] if i + 1 < len(catalogue_providers) else 'no more providers'
                update_status(0, f"Unexpected error occurred when connecting to {provider}. Trying {next_provider}",
                              update_status_func)
                # 捕获异常并打印详细错误信息，但避免红字输出
                print(f"An error occurred: {str(e)}")
                print("Continuing to the next provider...")

        if attempts == 0:
            update_status(0, "Failed to retrieve earthquake data on first attempt. Retrying in 60 seconds...",
                          update_status_func)
            time.sleep(60)  # Sleep 60 seconds before retry
        attempts += 1

    update_status(100, "Failed to retrieve earthquake data from all provided catalog sources after retry.",
                  update_status_func)
    print("-" * 50)
    return station, None, "Failed to download catalog data."


def download_station_data_logic(station, update_status_func=None, retries=2, overwrite=False):
    # 输出开始下载的日志
    print(f"Starting stream download for {station.report_date.strftime('%Y-%m-%d')}...")

    # 检查文件是否已经存在
    if station.check_existing_stream(overwrite=overwrite):
        return {"status": "exists", "message": f"Data for {station.report_date.strftime('%Y-%m-%d')} already exists."}

    # 初始化时间间隔
    start_time = station.report_date
    end_time = station.report_date + 86700
    duration = int((end_time - start_time) / 3)
    parts = []  # 用于存储每个部分的数据流

    # 下载三个部分的数据流
    for i in range(3):
        for attempt in range(retries + 1):
            try:
                if attempt == 0:
                    progress_message = f"Downloading part {i + 1}/3..."
                    progress_value = [5, 30, 60][i]
                    update_status(progress_value, progress_message, update_status_func)

                part = station.download_stream_part(start_time + i * duration,
                                                    start_time + (i + 1) * duration if i < 2 else end_time)
                parts.append(part)

                # 更新进度条到下一个步骤
                progress_value = [15, 50, 90][i]
                update_status(progress_value, f"Part {i + 1}/3 downloaded.", update_status_func)
                break

            except URLError as e:
                error_message = f"URLError while downloading part {i + 1}, attempt {attempt + 1} of {retries + 1}: {str(e)}"
                print(error_message)
                if attempt < retries:
                    retry_message = f"Attempt {attempt + 1} failed. Retrying in 15 seconds..."
                    print(retry_message)
                    time.sleep(15)
                else:
                    final_message = f"Failed to download part {i + 1} after {retries + 1} attempts."
                    print(final_message)
                    update_status(0, final_message, update_status_func)
                    return {"status": "fail", "message": final_message}

            except Exception as e:
                error_message = f"Error while downloading part {i + 1}, attempt {attempt + 1} of {retries + 1}: {str(e)}"
                print(error_message)
                if attempt < retries:
                    retry_message = f"Attempt {attempt + 1} failed. Retrying in 15 seconds..."
                    print(retry_message)
                    time.sleep(15)
                else:
                    final_message = f"Failed to download part {i + 1} after {retries + 1} attempts."
                    print(final_message)
                    update_status(0, final_message, update_status_func)
                    return {"status": "fail", "message": final_message}

    # 合并数据流
    if station.merge_streams(parts):
        print("Stream Downloaded.")  # 添加成功下载的日志输出
        station.stream.save_stream_as_miniseed(station, stream_to_save=station.stream.original_stream, identifier="")
        update_status(100, "All parts downloaded and merged successfully.", update_status_func)
        print("-" * 50)
        return {"status": "success", "message": "All parts downloaded and merged successfully."}
    else:
        error_message = "Stream merging failed."
        print(error_message)
        update_status(0, error_message, update_status_func)
        print("-" * 50)
        return {"status": "fail", "message": error_message}


def process_stream_logic(station, detrend_demean, detrend_linear, remove_outliers, apply_bandpass, taper, denoise,
                         update_status_func=None):
    # 输出开始处理的日志
    update_status(0, "Start Processing Stream...", update_status_func)

    if station.stream.original_stream is None:
        raise ValueError("Original stream is not set.")

    station.stream.processed_stream = station.stream.original_stream.copy()

    total_steps = sum([detrend_demean, detrend_linear, remove_outliers, apply_bandpass, taper, denoise])
    progress_increment = 100 / total_steps if total_steps > 0 else 100
    current_progress = 0

    if detrend_demean:
        update_status(current_progress, "Detrending (Demean)...", update_status_func)
        station.stream.process_detrend_demean()
        current_progress += progress_increment

    if detrend_linear:
        update_status(current_progress, "Detrending (Linear)...", update_status_func)
        station.stream.process_detrend_linear()
        current_progress += progress_increment

    if remove_outliers:
        update_status(current_progress, "Removing Outliers...", update_status_func)
        station.stream.process_remove_outliers()
        current_progress += progress_increment

    if apply_bandpass:
        update_status(current_progress, "Applying Bandpass Filter...", update_status_func)
        station.stream.process_apply_bandpass()
        current_progress += progress_increment

    if taper:
        update_status(current_progress, "Applying Taper...", update_status_func)
        station.stream.process_taper()
        current_progress += progress_increment

    if denoise:
        update_status(current_progress, "Denoising...", update_status_func)
        station.stream.process_denoise()
        current_progress += progress_increment

    update_status(100, "Processing Complete.", update_status_func)
    station.stream.save_stream_as_miniseed(station, stream_to_save=station.stream.processed_stream,
                                           identifier="processed")
    print("-" * 50)


def detect_phases_logic(station, p_threshold, s_threshold, update_status_func=None):
    update_status(10, "Starting phase detection...", update_status_func)
    update_status(20, "Predicting and annotating phases...", update_status_func)
    station.stream.predict_and_annotate()
    update_status(80, "Prediction and annotation complete.", update_status_func)
    update_status(85, "Filtering confidence levels...", update_status_func)
    station.stream.filter_confidence(p_threshold, s_threshold)

    p_count = sum(1 for pred in station.stream.picked_signals if pred['phase'] == 'P')
    s_count = sum(1 for pred in station.stream.picked_signals if pred['phase'] == 'S')

    update_status(100, f"Phase detection complete. P Waves detected: {p_count}, S Waves detected: {s_count}.",
                  update_status_func)

    station.stream.save_stream_as_miniseed(station, stream_to_save=station.stream.annotated_stream,
                                           identifier="processed.annotated")
    print("-" * 50)

    return station.stream.picked_signals, station.stream.annotated_stream, p_count, s_count


def match_events_logic(catalog, station, tolerance_p, tolerance_s, p_only, update_progress=None):
    # 开始匹配事件
    update_status(10, "Starting event matching...", update_progress)

    catalog.all_day_earthquakes = catalog.match_and_merge(
        station.stream.picked_signals,
        tolerance_p=tolerance_p,
        tolerance_s=tolerance_s,
        p_only=p_only
    )

    for eq in catalog.all_day_earthquakes:
        eq.update_errors()

    updated_df = catalog.update_summary_csv()

    # 计算 catalog 中的地震事件数量
    event_count = len(catalog.original_catalog_earthquakes)  # 使用原始 catalog 中的事件数

    # 直接使用 print_summary 方法获取结果
    detected_catalogued, detected_not_catalogued_count = catalog.print_summary()

    # 更新进度条到100%并完成匹配
    update_status(100, f"Matching complete. {detected_catalogued} out of {event_count} catalogued events detected.",
                  update_progress)

    print("-" * 50)

    return detected_catalogued, detected_not_catalogued_count, updated_df


def generate_report_logic(df, date_str, station_lat, station_lon, fill_map, simplified, p_only, update_status_func=None,
                          save_to_file=True):
    # 检查 DataFrame 是否为空
    if df.empty:
        update_status(0, "Error: No data available for report generation.", update_status_func)
        print("Error: DataFrame is empty. Report generation aborted.")
        return None

    date_str = date_str.strftime('%Y-%m-%d') if isinstance(date_str, datetime.date) else str(date_str)
    print("date_str", date_str)

    # Instantiate the DailyReport class
    update_status(10, "Generating catalogue plot...", update_status_func)
    daily_report = DailyReport(df, date_str, station_lat, station_lon, fill_map, simplified, p_only)

    # Plot the catalog and save the image
    daily_report.plot_catalogue()

    update_status(30, "Generating event plots...", update_status_func)

    # Generate and save event plots for all detected and catalogued earthquakes
    for _, row in df[(df['catalogued'] == True) & (df['detected'] == True) & (df['date'] == date_str)].iterrows():
        event_report = EventReport(row)
        # If simplified, do not plot confidence
        plot_path = event_report.plot_event(confidence=not simplified, p_only=daily_report.p_only)
        df.loc[df['unique_id'] == row['unique_id'], 'plot_path'] = plot_path

    update_status(60, "Creating report HTML...", update_status_func)
    # Assemble the HTML content for the daily report
    report_html = daily_report.assemble_daily_report_html()
    report_html = Report.convert_images_to_base64(report_html)

    update_status(80, "Creating PDF from HTML...", update_status_func)
    # Generate the PDF buffer from the HTML content
    pdf_buffer = daily_report.generate_pdf_buffer_from_html(report_html)

    if save_to_file:
        # Export the daily report as a PDF and save it to the appropriate directory
        pdf_file_path = daily_report.export_daily_report_pdf(report_html)
        update_status(100, "Report generation complete and saved to file.", update_status_func)
        print("-" * 50)
        return pdf_buffer
    else:
        update_status(100, "Report generation complete. PDF buffer created.", update_status_func)
        print("-" * 50)
        return pdf_buffer


def generate_event_report_logic(selected_eq, network, station_code, simplified=False, p_only=False):
    """
    Generate and provide a downloadable PDF report for the selected earthquake.

    Args:
    - selected_eq (pd.Series): The selected earthquake data as a pandas Series.
    - network (str): The network code for the station.
    - station_code (str): The station code.

    Returns:
    - pdf_buffer (io.BytesIO): The generated PDF report as a byte buffer.
    - file_name (str): The suggested file name for the PDF report.
    """
    # Create an EventReport object
    event_report = EventReport(selected_eq)

    # Generate the HTML content for the event report
    html_content = event_report.assemble_event_report_html(simplified, p_only)
    event_report_html = Report.convert_images_to_base64(html_content)

    # Generate a PDF buffer from the HTML content
    pdf_buffer = Report.generate_pdf_buffer_from_html(event_report_html)

    # Create a file name for the report
    unique_id = selected_eq['unique_id']
    file_name = f"earthquake_report_{network}_{station_code}_{unique_id}.pdf"

    return pdf_buffer, file_name


def send_email_logic(email_recipient, report_buffer, report_date):
    """
    Send the generated report via email.

    Args:
    - email_recipient (str): The recipient's email address.
    - report_buffer (BytesIO): The buffer containing the PDF report.
    - report_date (str): The date of the report to be used in the email and file name.

    Returns:
    - str: Status message indicating success or failure.
    """
    result = DailyReport.send_report_via_email(email_recipient, report_buffer, report_date)
    if result:
        return "Email sent successfully."
    else:
        return "Failed to send email."


def render_density_mapbox(df, station_info, title="Density Map of Detected Earthquakes"):
    """
    Render a density mapbox based on the filtered DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing the earthquake data.
    - station_info (dict): Dictionary containing station information (latitude, longitude, code).
    - title (str): Title of the density map plot.
    """
    st.subheader(title)

    # Copy the DataFrame to avoid SettingWithCopyWarning
    df_filtered = df.copy()

    # Ensure latitude and longitude are floats
    latitude = float(station_info["latitude"])
    longitude = float(station_info["longitude"])

    # Radio buttons for users to select the z-axis mapping option
    z_option = st.radio(
        "Color by",
        options=["Magnitude", "Depth"],
        index=0
    )

    # Determine the z axis based on the selected option
    if z_option == 'Magnitude':
        z_column = 'mag'
    else:  # Depth
        z_column = 'depth'

    # Create the density mapbox plot
    fig = px.density_mapbox(
        df_filtered,
        lat='lat',
        lon='long',
        z=z_column,  # Use the selected option for z-axis
        radius=10,
        center=dict(lat=latitude, lon=longitude),
        zoom=0,
        mapbox_style="open-street-map"
    )

    # Add the Bird plate boundaries to the map
    lines_lat, lines_lon = load_plate_boundaries()

    for lat, lon in zip(lines_lat, lines_lon):
        fig.add_trace(go.Scattermapbox(
            lat=lat,
            lon=lon,
            mode='lines',
            line=dict(width=1, color='#5B99C2'),
            showlegend=False,
            name='Plate Boundaries of Bird'
        ))

    # Update layout to match the style of render_interactive_map
    fig.update_layout(
        autosize=True,
        mapbox_style="light",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,  # Ensure that the legend is hidden
        mapbox_center={"lat": latitude, "lon": longitude},
        mapbox_accesstoken='pk.eyJ1IjoiZmFudGFzdGljbmFtZSIsImEiOiJjbHlnMnMzbmEwNmQ0MmpyN2lxNDNjaTd3In0.DfylrFmLO1EgfKf8sgIrkQ',
        coloraxis_colorbar=dict(
            title=z_option,  # Use the dynamic title
            titleside="bottom",
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickcolor='#000',
            lenmode="fraction",
            len=0.9,
            thicknessmode="pixels",
            thickness=15,
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.04
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def render_interactive_map(df, station_info, title="Detected Catalogued Earthquakes Map Plot"):
    """
    Render an interactive map based on the filtered DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing the earthquake data.
    - station_info (dict): Dictionary containing station information (latitude, longitude, code).
    - title (str): Title of the map plot.
    """
    st.subheader(title)

    # Create two columns for Color by and Size by radio buttons
    col1, col2 = st.columns(2)

    # Radio buttons for users to select the color mapping option
    with col1:
        color_by_option = st.radio(
            "Color by",
            options=["Magnitude", "Confidence", "Depth", "P Error"],
            index=0
        )

    # Radio buttons for users to select the size mapping option
    with col2:
        size_by_option = st.radio(
            "Size by",
            options=["Magnitude", "Confidence", "Depth", "P Error"],
            index=0
        )

    if df is not None and not df.empty:
        # Copy the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df.copy()

        # Ensure 'p_error' is a valid column in DataFrame and convert to absolute values
        if 'p_error' in df_filtered.columns:
            df_filtered['p_error'] = df_filtered['p_error'].abs()
        else:
            df_filtered['p_error'] = 0.0

        df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['time'])

        # Create a 'Location' column
        df_filtered['Location'] = df_filtered.apply(lambda row: f"{row['lat']:.2f}, {row['long']:.2f}", axis=1)

        # Combine magnitude and magnitude type into one column for tooltip
        df_filtered['Magnitude'] = df_filtered.apply(lambda row: f"{row['mag']} {row['mag_type']}", axis=1)

        # Create a 'Prediction Confidence' column as the maximum of p_confidence and s_confidence
        df_filtered['Prediction Confidence'] = df_filtered.apply(
            lambda row: row['p_confidence'] if pd.isnull(row['s_confidence']) else max(row['p_confidence'],
                                                                                       row['s_confidence']),
            axis=1
        )

        # Format 'time' column for tooltip
        df_filtered['formatted_time'] = df_filtered['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        successfully_detected = df_filtered[(df_filtered['catalogued'] == True) & (df_filtered['detected'] == True)]
        # Recalculate 'p_error' for successfully detected events
        successfully_detected['p_error'] = successfully_detected['p_error'].fillna(0).abs()

        # Determine the color axis based on the selected option
        if color_by_option == 'Magnitude':
            color_column = 'mag'
            tick_vals, tick_texts = create_ticks(df_filtered['mag'])
            coloraxis_title = "Magnitude"
        elif color_by_option == 'Confidence':
            color_column = 'Prediction Confidence'
            tick_vals, tick_texts = create_ticks(df_filtered['Prediction Confidence'])
            coloraxis_title = "Confidence"
        elif color_by_option == 'Depth':
            color_column = 'depth'
            tick_vals, tick_texts = create_ticks(df_filtered['depth'])
            coloraxis_title = "Depth (km)"
            tick_texts = [f"{tick} km" for tick in tick_texts]
        else:  # P Error
            color_column = 'p_error'
            tick_vals, tick_texts = create_ticks(df_filtered['p_error'])
            coloraxis_title = "P Error"

        # Determine the size axis based on the selected option
        if size_by_option == 'Magnitude':
            size_column = 'mag'
        elif size_by_option == 'Confidence':
            size_column = 'Prediction Confidence'
        elif size_by_option == 'Depth':
            size_column = 'depth'
        else:  # P Error
            size_column = 'p_error'

        fig = px.scatter_mapbox(
            successfully_detected,
            lat="lat",
            lon="long",
            size=size_column,  # Change size based on the selected option
            color=color_column,  # Change color based on the selected option
            hover_data={
                "Magnitude": True,
                "Location": True,
                "unique_id": True,
                "epi_distance": True,
                "depth": True,
                "Prediction Confidence": True,
                "P Error": successfully_detected['p_error']  # Show absolute p_error in hover data
            },
            color_continuous_scale=px.colors.sequential.Sunset,  # Changed color scale
            size_max=10,
            zoom=0
        )
        fig.update_traces(
            hovertemplate='<b>Date and Time:</b> %{customdata[0]}<br>' +
                          '<b>Magnitude:</b> %{customdata[1]}<br>' +
                          '<b>Location:</b> %{customdata[2]}<br>' +
                          '<b>Unique ID:</b> %{customdata[3]}<br>' +
                          '<b>Epicentral Distance (km):</b> %{customdata[4]:.2f}<br>' +
                          '<b>Depth (km):</b> %{customdata[5]}<br>' +
                          '<b>Prediction Confidence:</b> %{customdata[6]:.2f}<br>' +
                          '<b>P Error:</b> %{customdata[7]:.2f}<extra></extra>',  # Correctly reference absolute p_error
            customdata=successfully_detected[
                ['formatted_time', 'Magnitude', 'Location', 'unique_id', 'epi_distance', 'depth',
                 'Prediction Confidence', 'p_error']]  # Include P Error in custom data
        )

        # Add a station marker to the map
        add_station_marker(fig, station_info)

        # Add the Bird plate boundaries to the map without showing the legend
        lines_lat, lines_lon = load_plate_boundaries()

        for lat, lon in zip(lines_lat, lines_lon):
            fig.add_trace(go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode='lines',
                line=dict(width=1, color='#5B99C2'),
                showlegend=False,
                name='Plate Boundaries of Bird'
            ))

        # Update layout to include the unified color axis legend
        fig.update_layout(
            autosize=True,
            mapbox_style="light",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=False,  # Ensure that the legend is hidden
            mapbox_center={"lat": successfully_detected['lat'].mean(), "lon": successfully_detected['long'].mean()},
            mapbox_accesstoken='pk.eyJ1IjoiZmFudGFzdGljbmFtZSIsImEiOiJjbHlnMnMzbmEwNmQ0MmpyN2lxNDNjaTd3In0.DfylrFmLO1EgfKf8sgIrkQ',
            coloraxis_colorbar=dict(
                title=coloraxis_title,  # Use the dynamic title
                titleside="bottom",
                ticks="outside",
                ticklen=4,
                tickwidth=1,
                tickcolor='#000',
                tickvals=tick_vals,
                ticktext=tick_texts,
                lenmode="fraction",
                len=0.9,
                thicknessmode="pixels",
                thickness=15,
                yanchor="middle",
                y=0.5,
                xanchor="right",
                x=1.04
            )
        )

        selected_point = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

        # Handle point selection and further actions
        handle_point_selection(selected_point, df_filtered)


def handle_point_selection(selected_point, df_filtered):
    """
    Handle the selection of a point on the map and trigger the appropriate actions.

    Args:
    - selected_point: The selected point on the map.
    - df_filtered (pd.DataFrame): The DataFrame containing the earthquake data.

    Returns:
    - None: The function handles the selection and triggers further actions like displaying details and generating reports.
    """
    # Check if any point is selected and has "points" data
    if selected_point and selected_point.selection and "points" in selected_point.selection:
        points = selected_point.selection["points"]
        if points:  # Ensure there is at least one point selected
            unique_id = points[0]["customdata"][3]

            # Look up plot_path and other details using unique_id in the DataFrame
            selected_eq = df_filtered[df_filtered["unique_id"] == unique_id].iloc[0]

            # Generate and display detailed earthquake information
            earthquake_info_html = EventReport.event_details_html(selected_eq,
                                                                  st.session_state.get('simplified', 'N/A'),
                                                                  st.session_state.get('p_only', 'N/A'))
            earthquake_info_html = Report.convert_images_to_base64(earthquake_info_html)

            st.markdown(earthquake_info_html, unsafe_allow_html=True)

            # Directly trigger the PDF download
            pdf_buffer, file_name = generate_event_report_logic(selected_eq, st.session_state.get('network', 'N/A'),
                                                                st.session_state.get('station_code', 'N/A'),
                                                                st.session_state.get('simplified', 'N/A'),
                                                                st.session_state.get('p_only', 'N/A'))

            st.download_button(
                label="Download Report PDF For Selected Event",
                data=pdf_buffer,
                file_name=f"earthquake_report_{unique_id}.pdf",
                mime="application/pdf",
            )


def add_station_marker(fig, station_info):
    """
    Add a station marker to the provided Plotly figure.

    Args:
    - fig (go.Figure): The Plotly figure to which the station marker will be added.
    - station_info (dict): Dictionary containing station information (latitude, longitude, code).

    Returns:
    - None: The function modifies the figure in place by adding the station marker.
    """
    if station_info:
        station_data = {
            'latitude': [station_info['latitude']],
            'longitude': [station_info['longitude']],
        }
        df_station = pd.DataFrame(station_data)

        station_trace = go.Scattermapbox(
            lat=df_station['latitude'],
            lon=df_station['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=15,
                symbol='triangle'
            ),
            hovertemplate=f'<b>Station:</b> {station_info["code"]}<extra></extra>',
            name=''
        )
        fig.add_trace(station_trace)


def daily_report_automation_logic():
    # 获取当天日期
    current_date = datetime.datetime.now() - datetime.timedelta(days=1)
    default_config = load_config_to_df()
    report_date_str = current_date.strftime('%Y-%m-%d')

    # 状态信息块
    with st.status(f"Processing data for {report_date_str}"):
        time.sleep(1)

        # Step 1: Catalog Data Settings
        network = default_config['network']
        station_code = default_config['station_code']
        data_provider_url = default_config['data_provider_url']
        latitude = default_config['station_latitude']
        longitude = default_config['station_longitude']
        catalog_providers = default_config['catalog_providers'].split(', ')
        radmin = float(default_config['radmin'])
        radmax = float(default_config['radmax'])
        minmag = float(default_config['minmag'])
        maxmag = float(default_config['maxmag'])

        st.write("Searching for catalog data...")
        time.sleep(2)
        st.write("Downloading catalog data...")
        time.sleep(1)

        # Download catalog data
        station, catalog, message = download_catalogue_logic(
            network,
            station_code,
            data_provider_url,
            current_date,
            latitude,
            longitude,
            radmin,
            radmax,
            minmag,
            maxmag,
            catalog_providers
        )

        if not catalog:
            st.write(f"Failed to download catalog data: {message}")
            return
        else:
            st.write(f"Catalog downloaded successfully. Number of events: {len(catalog.original_catalog_earthquakes)}")

        st.write("Downloading station data...")
        time.sleep(1)

        # Step 2: Station Data Settings
        overwrite = default_config['overwrite']

        # Download station data
        result = download_station_data_logic(station, overwrite=overwrite)

        if result['status'] == 'success' or result['status'] == 'exists':
            st.write("Station data downloaded successfully.")

            # Step 3: Process Stream Data
            detrend_demean = default_config['detrend_demean']
            detrend_linear = default_config['detrend_linear']
            remove_outliers = default_config['remove_outliers']
            apply_bandpass = default_config['apply_bandpass']
            taper = default_config['taper']
            denoise = default_config['denoise']

            st.write("Processing stream data...")
            time.sleep(1)

            # Process stream data
            process_stream_logic(station, detrend_demean, detrend_linear, remove_outliers, apply_bandpass, taper,
                                 denoise)
            st.write("Stream processing completed and saved.")

            # Step 4: Detect Phases
            p_threshold = default_config['p_threshold']
            s_threshold = default_config['s_threshold']

            st.write("Detecting phases...")
            time.sleep(1)

            # Detect phases
            picked_signals, annotated_stream, p_count, s_count = detect_phases_logic(
                station, p_threshold, s_threshold)
            st.write(f"P waves detected: {p_count}, S waves detected: {s_count}")

            # Step 5: Match Events
            tolerance_p = default_config['tolerance_p']
            tolerance_s = default_config['tolerance_s']
            p_only = default_config['p_only']

            st.write("Matching events...")
            time.sleep(1)

            # Match events
            detected_catalogued, detected_not_catalogued_count, updated_df = match_events_logic(
                catalog, station, tolerance_p, tolerance_s, p_only)
            st.write(
                f"Detected Catalogued Events: {detected_catalogued}, Detected Not Catalogued Count: {detected_not_catalogued_count}")

            # Only proceed with report generation and email if there are detected catalogued events
            if detected_catalogued > 0:
                # Step 6: Generate Report
                simplified = default_config['simplified']
                fill_map = default_config['fill_map']

                st.write("Generating report...")
                time.sleep(1)

                pdf_buffer = generate_report_logic(
                    updated_df, report_date_str,
                    latitude, longitude, fill_map, simplified, p_only, save_to_file=True)

                st.write("Report generated.")

                # Step 7: Send Email
                st.write("Sending email...")
                email_result = send_email_logic(default_config['email_recipient'], pdf_buffer, report_date_str)
                st.write(email_result)

            else:
                st.write("No catalogued events detected, skipping report generation and email.")

        else:
            st.write(f"Failed to download station data for {report_date_str}: {result['message']}")

    st.button("Rerun")


def initialisation_logic(network, station_code, data_provider_url, email_recipient, max_attempts=3):
    """
    Handles the initialization logic including fetching coordinates, creating summary file,
    saving configuration, and reloading the application with progress updates.

    Args:
    - network (str): The network name.
    - station_code (str): The station code.
    - data_provider_url (str): The data provider URL.
    - email_recipient (str): The email address for receiving reports.
    - max_attempts (int): Maximum number of attempts to fetch coordinates.

    Returns:
    - None
    """
    progress = st.progress(0)
    status_text = st.empty()

    status_text.text("Starting initialization...")
    progress.progress(10)

    # Attempt to fetch coordinates with retries
    latitude, longitude = None, None
    for attempt in range(1, max_attempts + 1):
        status_text.text(f"Fetch Station Coordinates (Attempt {attempt}/{max_attempts})")
        latitude, longitude = Station.fetch_coordinates(network, station_code, data_provider_url)

        if latitude is not None and longitude is not None:
            status_text.text("Coordinates fetched successfully.")
            progress.progress(30)
            break
        else:
            status_text.text(f"Failed to Fetch Station Coordinates, Trying again (Attempt {attempt}/{max_attempts})")
            time.sleep(2)

    if latitude is None or longitude is None:
        status_text.text("Failed to fetch coordinates after multiple attempts. Initialization failed.")
        progress.progress(100)
        return

    # Update session state with new settings
    st.session_state['network'] = network
    st.session_state['station_code'] = station_code
    st.session_state['data_provider_url'] = data_provider_url
    st.session_state['station_latitude'] = str(latitude)
    st.session_state['station_longitude'] = str(longitude)
    st.session_state['email_recipient'] = email_recipient
    st.session_state['initialized'] = True

    status_text.text("Session state updated.")
    progress.progress(50)

    # Load and update config file to reflect changes
    config_file = load_config_to_df()
    config_file['network'] = network
    config_file['station_code'] = station_code
    config_file['data_provider_url'] = data_provider_url
    config_file['station_latitude'] = str(latitude)
    config_file['station_longitude'] = str(longitude)
    config_file['email_recipient'] = email_recipient
    config_file['initialized'] = True

    # Save the updated config to file
    save_config_to_yaml(config_file)

    status_text.text("Configuration saved.")
    progress.progress(70)

    # Create an empty summary file
    initialize_summary_file(network, station_code, deployed=True)
    status_text.text("Empty summary file created.")
    progress.progress(90)

    # Display success message and prepare for reload
    status_text.text("Settings saved! Now the application will reload for you to continue.")
    progress.progress(100)
    time.sleep(2)
    st.rerun()  # Reload the application
