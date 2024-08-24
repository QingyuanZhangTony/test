import datetime
import time

from logic import download_station_data_logic, download_catalogue_logic, process_stream_logic, detect_phases_logic, \
    match_events_logic, generate_report_logic, send_email_logic
from streamlit_utils import load_config_to_df, read_summary_csv

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # Load default settings
    default_config = load_config_to_df()

    # Calculate the date range from July 1st of the current year to yesterday
    start_date = datetime.date(2024, 7, 6)
    end_date = datetime.date.today() - datetime.timedelta(days=1)

    current_date = start_date
    while current_date <= end_date:
        report_date_str = current_date.strftime('%Y-%m-%d')
        print(f"Processing data for {report_date_str}")

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
            print(f"Failed to download catalog data: {message}")
        else:
            print(f"Catalog downloaded successfully. Number of events: {len(catalog.original_catalog_earthquakes)}")

            # Step 2: Station Data Settings
            overwrite = default_config['overwrite']

            # Download station data
            result = download_station_data_logic(
                station,
                overwrite= overwrite
            )

            if result['status'] == 'success' or result['status'] == 'exists':
                print("Station data downloaded successfully.")

                # Step 3: Process Stream Data
                detrend_demean = default_config['detrend_demean']
                detrend_linear = default_config['detrend_linear']
                remove_outliers = default_config['remove_outliers']
                apply_bandpass = default_config['apply_bandpass']
                taper = default_config['taper']
                denoise = default_config['denoise']

                # Process stream data
                process_stream_logic(station, detrend_demean, detrend_linear, remove_outliers, apply_bandpass, taper,
                                     denoise)
                print("Stream processing completed and saved.")

                # Step 4: Detect Phases
                p_threshold = default_config['p_threshold']
                s_threshold = default_config['s_threshold']

                # Detect phases
                picked_signals, annotated_stream, p_count, s_count = detect_phases_logic(
                    station, p_threshold, s_threshold)
                print(f"P waves detected: {p_count}, S waves detected: {s_count}")

                # Step 5: Match Events
                tolerance_p = default_config['tolerance_p']
                tolerance_s = default_config['tolerance_s']
                p_only = default_config['p_only']

                # Match events
                detected_catalogued, detected_not_catalogued_count = match_events_logic(
                    catalog, station, tolerance_p, tolerance_s, p_only)
                print(
                    f"Detected Catalogued Events: {detected_catalogued}, Detected Not Catalogued Count: {detected_not_catalogued_count}")

                # Step 6: Generate Report
                simplified = default_config['simplified']
                fill_map = default_config['fill_map']

                # Read summary CSV and generate the report PDF
                df = read_summary_csv(network, station_code)
                filtered_df = df[df['date'] == report_date_str]

                pdf_buffer = generate_report_logic(
                    filtered_df, report_date_str,
                    latitude, longitude, fill_map, simplified, p_only, save_to_file=True)



                print("Report generated.")

                # Step 7: Send Email
                #email_result = send_email_logic(default_config['email_recipient'], pdf_buffer, report_date_str)
                #print(email_result)

            else:
                print(f"Failed to download station data for {report_date_str}: {result['message']}")

        # Move to the next day regardless of success or failure
        current_date += datetime.timedelta(days=1)
        print('Waiting 60 seconds before processing the next day...')
        time.sleep(60)
