import datetime
import os

import streamlit as st

from logic import download_station_data_logic, download_catalogue_logic, process_stream_logic, detect_phases_logic, \
    match_events_logic, generate_report_logic, send_email_logic
from streamlit_utils import load_config_to_df, read_summary_csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


import time
import streamlit as st

def daily_report_automation_for_yesterday():
    # 获取当天日期
    current_date = datetime.datetime.now() - datetime.timedelta(days=2)

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
        result = download_station_data_logic(
            station,
            overwrite=overwrite
        )

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
            detected_catalogued, detected_not_catalogued_count = match_events_logic(
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

                # Read summary CSV and generate the report PDF
                df, status = read_summary_csv(network, station_code)
                filtered_df = df[df['date'] == report_date_str]

                pdf_buffer = generate_report_logic(
                    filtered_df, report_date_str,
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


# Streamlit 页面
st.title("Automated Earthquake Data Processing")

# 在页面加载时自动运行处理逻辑
daily_report_automation_for_yesterday()
