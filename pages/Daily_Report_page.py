import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from logic import download_station_data_logic, download_catalogue_logic, process_stream_logic, detect_phases_logic, \
    match_events_logic, generate_report_logic, send_email_logic, render_interactive_map
from streamlit_utils import load_config_to_session, initialize_state_defaults, load_summary_to_session, update_status, \
    sidebar_navigation
import os
import warnings




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load default settings
load_config_to_session()

# Initialize state variables
initialize_state_defaults()

warnings.simplefilter(action='ignore', category=FutureWarning)

# Sidebar navigation
sidebar_navigation()

# Application Title
st.title("Daily Earthquake Identification")

# Reset Container
reset_container = st.container()
with reset_container:
    if st.button('Reset Session', key='reset_button'):
        # Clear all session state
        for key in st.session_state.keys():
            del st.session_state[key]
        # Reinitialize the state defaults
        initialize_state_defaults()
        load_config_to_session()
        st.rerun()

# Catalog Data Container
catalog_settings_container = st.container()
with catalog_settings_container:
    catalog_help_text = "This step requests earthquake catalog from the specified providers."
    st.header("Step 1: Downloading Earthquake Catalog", help=catalog_help_text)

    report_date = st.date_input("Date For Downloading Catalog:",
                                value=datetime.date.today() - datetime.timedelta(days=1))
    st.session_state.report_date = report_date

    if 'catalog_download_progress' not in st.session_state:
        st.session_state.catalog_download_progress = 0
        st.session_state.catalog_download_status = "Ready to download"

    progress_bar = st.progress(st.session_state.catalog_download_progress)
    status_text = st.text(st.session_state.catalog_download_status)

    proceed_button_catalog = st.button(
        'Download Catalog' if not st.session_state.is_catalog_downloading else 'Downloading...',
        key=f"proceed_catalog_{st.session_state.is_catalog_downloading}",
        disabled=st.session_state.is_catalog_downloading)

    if proceed_button_catalog or st.session_state.is_catalog_downloading:
        st.session_state.is_catalog_downloading = True
        report_date_str = st.session_state.report_date.strftime('%Y-%m-%d')


        def update_progress(progress_value, status_message):
            st.session_state.catalog_download_progress = progress_value
            st.session_state.catalog_download_status = status_message
            progress_bar.progress(progress_value / 100.0)
            status_text.text(status_message)


        station, catalog, message = download_catalogue_logic(
            st.session_state.network,
            st.session_state.station_code,
            st.session_state.data_provider_url,
            st.session_state.report_date,
            st.session_state.station_latitude,
            st.session_state.station_longitude,
            float(st.session_state.radmin),
            float(st.session_state.radmax),
            float(st.session_state.minmag),
            float(st.session_state.maxmag),
            [provider.strip() for provider in st.session_state.catalog_providers.split(',')],
            update_status_func=update_progress
        )

        st.session_state.global_station = station

        if catalog:
            st.session_state.global_catalog = catalog
            st.session_state.catalog_downloaded = True
            st.session_state.event_count = len(catalog.original_catalog_earthquakes)
            st.session_state.catalog_download_status = message
            st.session_state.catalog_provider = catalog.provider
        else:
            st.session_state.catalog_downloaded = False
            st.session_state.catalog_download_status = message
            st.session_state.event_count = 0
            st.session_state.catalog_provider = "Not Downloaded"

        st.session_state.is_catalog_downloading = False
        st.session_state.catalog_download_progress = 100
        st.rerun()

    if st.session_state.catalog_downloaded:
        with st.expander("Click Here To See Details Of Catalog", expanded=False):
            st.subheader("Catalog Earthquake Details")
            st.markdown("##### Map Plot")

            col1, col2 = st.columns([1, 3])
            with col1:
                projection_type = st.radio("Choose Projection Type", ['Local', 'Orthographic', 'Global'],
                                           horizontal=True)

                projection_map = {'Local': 'local', 'Orthographic': 'ortho', 'Global': 'global'}
                selected_projection = projection_map[projection_type]

            if 'catalog_static_plot_buf' not in st.session_state or st.session_state.selected_projection != selected_projection:
                st.session_state.selected_projection = selected_projection
                catalog_static_plot = BytesIO()
                st.session_state.global_catalog.original_catalog.plot(projection=selected_projection,
                                                                      outfile=catalog_static_plot, format='png')
                catalog_static_plot.seek(0)
                st.session_state.catalog_static_plot_buf = catalog_static_plot

            if 'catalog_static_plot_buf' in st.session_state:
                st.image(st.session_state.catalog_static_plot_buf, caption=f"{projection_type} Projection")

            earthquake_data = [
                {
                    'Time': eq.time.datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'Location': f"{eq.lat:.2f}, {eq.long:.2f}",
                    'Magnitude': f"{eq.mag} {eq.mag_type}",
                    'Depth (km)': f"{eq.depth:.2f} km",
                    'Epicentral Distance (km)': f"{eq.epi_distance:.2f} km" if eq.epi_distance is not None else 'N/A'
                }
                for eq in st.session_state.global_catalog.original_catalog_earthquakes
            ]

            earthquake_df = pd.DataFrame(earthquake_data)
            st.markdown("##### Details")
            st.dataframe(earthquake_df)
st.divider()

# Station Data Container
station_settings_container = st.container()
with station_settings_container:
    station_help_text = "This step downloads stream data and fetches station location."
    st.header("Step 2: Download Stream Data From Station", help=station_help_text)

    if 'station_download_progress' not in st.session_state:
        st.session_state.station_download_progress = 0
        st.session_state.station_download_status = "Ready to download"

    progress_bar = st.progress(st.session_state.station_download_progress)
    status_text = st.text(st.session_state.station_download_status)

    proceed_button = st.button('Download Station Data' if not st.session_state.is_downloading else 'Downloading...',
                               key='proceed_station', disabled=st.session_state.is_downloading)

    if proceed_button or st.session_state.is_downloading:
        st.session_state.is_downloading = True
        report_date_str = st.session_state.report_date.strftime('%Y-%m-%d')


        def update_progress(progress_value, status_message):
            st.session_state.station_download_progress = progress_value
            st.session_state.station_download_status = status_message
            progress_bar.progress(progress_value / 100.0)
            status_text.text(status_message)


        result = download_station_data_logic(
            station=st.session_state.global_station,
            update_status_func=update_progress,
            retries=2,
            overwrite=st.session_state.overwrite
        )

        if result['status'] == 'success':
            st.session_state.station_downloaded = True
            st.session_state.station_download_status = f"Stream downloaded for {report_date_str}."
        elif result['status'] == 'exists':
            st.session_state.station_downloaded = True
            st.session_state.station_download_status = result['message']
        else:
            st.session_state.station_downloaded = False
            st.session_state.station_download_status = result['message']

        st.session_state.is_downloading = False
        st.session_state.station_download_progress = 100 if st.session_state.station_downloaded else 0
        st.rerun()

    if st.session_state.station_downloaded and not st.session_state.is_downloading:
        with st.expander("Click Here To See The Waveform Plot", expanded=False):
            st.subheader("Waveform Plot Of Downloaded Stream")
            original_stream_plot = BytesIO()

            st.session_state.global_station.stream.original_stream.plot(outfile=original_stream_plot, format='png')

            original_stream_plot.seek(0)
            st.image(original_stream_plot)
            st.session_state.original_stream_plot_buf = original_stream_plot
st.divider()

# Process Stream Container
process_stream_container = st.container()
with process_stream_container:
    process_help_text = "This step performs stream signal processing."
    st.header("Step 3: Process Stream Data", help=process_help_text)

    if 'stream_process_progress' not in st.session_state:
        st.session_state.stream_process_progress = 0
        st.session_state.stream_process_status = "Ready to process"

    progress_bar = st.progress(st.session_state.stream_process_progress)
    status_text = st.text(st.session_state.stream_process_status)

    proceed_button_process = st.button('Process Stream' if not st.session_state.is_processing else 'Processing...',
                                       key='proceed_process', disabled=st.session_state.is_processing)

    if proceed_button_process and st.session_state.global_station:
        st.session_state.is_processing = True
        st.rerun()

    if st.session_state.is_processing:
        def update_progress(percent, status):
            st.session_state.stream_process_progress = percent
            st.session_state.stream_process_status = status
            progress_bar.progress(percent / 100.0)
            status_text.text(status)


        process_stream_logic(
            st.session_state.global_station,
            st.session_state.detrend_demean,
            st.session_state.detrend_linear,
            st.session_state.remove_outliers,
            st.session_state.apply_bandpass,
            st.session_state.taper,
            st.session_state.denoise,
            update_status_func=update_progress
        )

        st.session_state.stream_processed = True

        img_buf_processed = BytesIO()
        st.session_state.global_station.stream.processed_stream.plot(outfile=img_buf_processed, format='png')
        img_buf_processed.seek(0)
        st.session_state.img_buf_processed = img_buf_processed

        st.session_state.is_processing = False
        st.session_state.stream_process_progress = 100
        st.session_state.stream_process_status = "Processing Complete"
        st.rerun()

    if 'img_buf_processed' in st.session_state:
        with st.expander("Click Here To See The Waveform Plot", expanded=False):
            st.subheader("Waveform Plot Of Stream After Processing")
            st.image(st.session_state.img_buf_processed)
st.divider()

# Detect Phases Container
detect_phases_container = st.container()
with detect_phases_container:
    detect_phases_help_text = "This step will detect P/S waves from the stream. CUDA will be utilized for phase picking if available."
    st.header("Step 4: Detect Phases", help=detect_phases_help_text)

    if 'phase_detection_progress' not in st.session_state:
        st.session_state.phase_detection_progress = 0
        st.session_state.phase_detection_status = "Ready to detect phases"

    progress_bar = st.progress(st.session_state.phase_detection_progress)
    status_text = st.text(st.session_state.phase_detection_status)

    proceed_button_detect = st.button('Proceed with Detection' if not st.session_state.is_detecting else 'Detecting...',
                                      key='proceed_detect', disabled=st.session_state.is_detecting)

    if proceed_button_detect and st.session_state.global_station:
        st.session_state.is_detecting = True
        st.rerun()

    if st.session_state.is_detecting:
        def update_progress(progress_value, status_message):
            st.session_state.phase_detection_progress = progress_value
            st.session_state.phase_detection_status = status_message
            progress_bar.progress(progress_value / 100.0)
            status_text.text(status_message)


        picked_signals, annotated_stream, p_count, s_count = detect_phases_logic(
            st.session_state.global_station,
            st.session_state.p_threshold,
            st.session_state.s_threshold,
            update_status_func=update_progress
        )

        st.session_state.phases_detected = True
        st.session_state.p_count = p_count
        st.session_state.s_count = s_count

        st.session_state.is_detecting = False
        st.session_state.phase_detection_progress = 100
        st.rerun()
st.divider()

# Match Events Container
match_events_container = st.container()
with match_events_container:
    match_events_help_text = "This step matches detected P/S waves with the earthquakes from the catalog."
    st.header("Step 5: Match Events", help=match_events_help_text)

    # Initialize progress bar and status text if not already done
    if 'match_events_progress' not in st.session_state:
        st.session_state.match_events_progress = 0
        st.session_state.match_events_status = "Ready to match events"

    # Display the progress bar and status text
    progress_bar = st.progress(st.session_state.match_events_progress)
    status_text = st.text(st.session_state.match_events_status)

    # Create two columns with a 3:2 ratio
    col1, col2 = st.columns([2, 4])

    with col1:
        if 'is_matching' not in st.session_state:
            st.session_state.is_matching = False

        # Create the Match button
        proceed_button_match = st.button('Proceed with Matching' if not st.session_state.is_matching else 'Matching...',
                                         key='proceed_match', disabled=st.session_state.is_matching)

    # Check if the match button was pressed and necessary data is available
    if proceed_button_match and st.session_state.global_catalog and st.session_state.phases_detected:
        # Step 1: Start the matching process
        st.session_state.is_matching = True
        st.rerun()

    if st.session_state.is_matching:
        # Step 2: Define progress update function
        def update_progress(progress_value, status_message):
            st.session_state.match_events_progress = progress_value
            st.session_state.match_events_status = status_message
            progress_bar.progress(progress_value / 100.0)
            status_text.text(status_message)

        # Step 3: Reset matching state before starting
        st.session_state.matching_completed = False
        st.session_state.detected_catalogued = 0

        # Step 4: Perform the event matching logic
        detected_catalogued, detected_not_catalogued_count, updated_df = match_events_logic(
            st.session_state.global_catalog,
            st.session_state.global_station,
            st.session_state.tolerance_p,
            st.session_state.tolerance_s if 'tolerance_s' in st.session_state else 0.0,
            st.session_state.p_only,
            update_progress=update_progress  # 传递进度更新函数
        )

        # Step 5: Update status and data after matching
        st.session_state.matching_completed = True
        st.session_state.detected_catalogued = detected_catalogued
        st.session_state.df = updated_df

        # Step 6: Matching complete
        st.session_state.is_matching = False
        st.rerun()


# Generate Report Container
generate_report_container = st.container()
with generate_report_container:
    report_help_text = "This step generates a detailed report for successfully matched earthquakes."
    st.header("Step 6: Generate Report", help=report_help_text)

    # Step 1: Initialize the report generation state
    if 'is_generating_report' not in st.session_state:
        st.session_state.is_generating_report = False

    if 'global_report_buffer' not in st.session_state:
        st.session_state.global_report_buffer = None  # 初始化为 None

    if 'report_generation_progress' not in st.session_state:
        st.session_state.report_generation_progress = 0

    # Step 2: Prepare UI elements for progress and status display
    status_text = st.empty()
    progress_bar = st.progress(st.session_state.report_generation_progress)

    report_date_str = st.session_state.report_date.strftime('%Y-%m-%d')

    # Step 3: Filter data for the selected report date
    report_date_df = st.session_state.df[st.session_state.df['date'] == report_date_str]




    if report_date_df.empty:
        st.warning("No data available for the selected date. Please select a different date.")
        st.stop()

    # Step 4: Ensure matching is completed and detected events exist
    if st.session_state.get('matching_completed', False) and 'detected_catalogued' in st.session_state:
        if st.session_state.detected_catalogued < 1:
            st.warning("No event matched. Please confirm before generating the report.")
    else:
        st.warning("Event matching not completed. Please complete the event matching step before generating the report.")
        st.stop()

    # Step 5: Handle the report generation button press
    proceed_button_report = st.button(
        'Generate Report' if not st.session_state.is_generating_report else 'Generating...',
        key='proceed_report', disabled=st.session_state.is_generating_report)

    if proceed_button_report:
        # Step 6: Start report generation
        st.session_state.is_generating_report = True
        st.session_state.report_generation_progress = 10
        st.session_state.report_generation_status = "Starting report generation..."
        st.rerun()

    if st.session_state.is_generating_report:
        # Step 7: Define progress update function
        def update_progress(progress_value, status_message):
            st.session_state.report_generation_progress = progress_value
            st.session_state.report_generation_status = status_message
            status_text.text(status_message)
            progress_bar.progress(progress_value / 100.0)

        # Step 8: Verify all required data is available before generating the report
        if not st.session_state.global_station or not st.session_state.global_catalog or not st.session_state.matching_completed:
            st.session_state.report_generation_status = "Required data not available. Please complete all steps."
            st.session_state.is_generating_report = False
        else:
            # Step 9: Load summary and generate the report
            st.session_state.global_report_buffer = generate_report_logic(
                st.session_state.df,
                st.session_state.report_date,
                st.session_state.global_station.latitude,
                st.session_state.global_station.longitude,
                st.session_state.fill_map,
                st.session_state.simplified,
                st.session_state.p_only,
                update_status_func=update_progress,
                save_to_file=True  # Return buffer instead of saving to file
            )

            # Step 10: Finalize the report generation process
            st.session_state.report_generated = True
            st.session_state.is_generating_report = False
            st.session_state.report_generation_progress = 100
            st.session_state.report_generation_status = "Report generation complete."
            st.rerun()

    # Step 11: Display report generation status
    if st.session_state.report_generated:
        status_text.text(st.session_state.report_generation_status)

    # Step 12: Optionally display an interactive map
    if st.session_state.detected_catalogued > 0:
        with st.expander("Click Here To See Detected Earthquakes On An Interactive Map", expanded=False):
            render_interactive_map(report_date_df,
                                   station_info={
                                       "latitude": st.session_state.station_latitude,
                                       "longitude": st.session_state.station_longitude,
                                       "code": st.session_state.station_code
                                   },
                                   title="Detected Earthquakes on Report Date")
st.divider()


# Send Email Container
send_email_container = st.container()
with send_email_container:
    email_help_text = 'This step will send the generated report via Email.'
    st.header("Step 7: Send Report", help=email_help_text)

    if 'is_sending_email' not in st.session_state:
        st.session_state.is_sending_email = False

    if 'email_send_progress' not in st.session_state:
        st.session_state.email_send_progress = 0
        st.session_state.email_send_status = "Ready to send email"

    # Display the progress bar and status text
    status_text = st.empty()
    progress_bar = st.progress(st.session_state.email_send_progress)

    col1, col2 = st.columns([2, 4])

    with col1:
        proceed_button_email = st.button('Send Report via Email' if not st.session_state.is_sending_email else 'Sending...',
                                         key='proceed_email', disabled=st.session_state.is_sending_email)

        # Add Download Report PDF button
        if st.session_state.report_generated and st.session_state.global_report_buffer:
            st.download_button(
                label="Download Report PDF Directly",
                data=st.session_state.global_report_buffer,
                file_name=f"daily_report_{st.session_state.report_date.strftime('%Y-%m-%d')}.pdf",
                mime="application/pdf",
            )

    if proceed_button_email and st.session_state.global_report_buffer:
        st.session_state.is_sending_email = True
        st.session_state.email_send_progress = 10
        st.session_state.email_send_status = "Starting email sending process..."
        st.rerun()  # Refresh page to show "Sending..." and disable button

    if st.session_state.is_sending_email:
        def update_progress(progress_value, status_message):
            st.session_state.email_send_progress = progress_value
            st.session_state.email_send_status = status_message
            status_text.text(status_message)
            progress_bar.progress(progress_value / 100.0)


        update_progress(30, "Preparing to send email...")

        # Send the email using the send_email_logic function
        email_result = send_email_logic(
            st.session_state.email_recipient,
            st.session_state.global_report_buffer,
            st.session_state.report_date.strftime('%Y-%m-%d')
        )

        if email_result == "Email sent successfully.":
            st.session_state.email_sent = True
            update_progress(100, "Email sent successfully.")
        else:
            st.session_state.email_sent = False
            update_progress(100, "Failed to send email. Please try again.")

        st.session_state.is_sending_email = False

        # Refresh the page only if needed (e.g., to reset the button status)
        if st.session_state.email_send_progress == 100:
            st.rerun()  # Refresh page to restore button text and enable button
st.divider()
