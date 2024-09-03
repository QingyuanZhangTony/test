import datetime
import os
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from github import GithubException
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from obspy.core import UTCDateTime
from github_file import upload_file_to_github, check_file_exists_in_github, write_file_to_github, g, repo, \
    download_file_from_github, repo, REPO_NAME, get_file_sha, move_github_file
from earthquake import Earthquake

from github.ContentFile import ContentFile

from streamlit_utils import read_summary_csv


class CatalogData:
    def __init__(self, station, radmin, radmax, minmag, maxmag, catalogue_providers):
        self.radmin = radmin
        self.radmax = radmax
        self.minmag = minmag
        self.maxmag = maxmag
        self.catalogue_providers = catalogue_providers

        self.station = station
        self.latitude = station.latitude
        self.longitude = station.longitude
        self.date = station.report_date

        self.provider = None
        self.event_counter = 1

        self.original_catalog = None
        self.original_catalog_earthquakes = []
        self.all_day_earthquakes = []

        self.catalog_plot_path = None

    def __str__(self):
        if self.all_day_earthquakes and self.provider:
            return f"Catalog retrieved from {self.provider}. {len(self.all_day_earthquakes)} earthquakes found."
        else:
            return "No catalog data available."

    # In the Catalog class (catalog.py)

    def request_catalogue_from_provider(self, provider):
        """
        Request the earthquake catalog from a single provider.

        Parameters:
        -----------
        provider : str
            The URL of the earthquake catalog provider.

        Returns:
        --------
        Catalog : obspy.core.event.Catalog or None
            The catalog object containing earthquake events if successful, or None if no data was retrieved.
        """

        client = Client(provider)
        starttime = self.station.report_date - 30 * 5  # 5 minutes before midnight on the day before
        endtime = self.station.report_date + (24 * 3600) + 30 * 5  # 5 minutes after midnight on the day after

        catalog = client.get_events(
            latitude=self.station.latitude,
            longitude=self.station.longitude,
            minradius=self.radmin,
            maxradius=self.radmax,
            starttime=starttime,
            endtime=endtime,
            minmagnitude=self.minmag,
            maxmagnitude=self.maxmag
        )

        if catalog.events:
            self.provider = provider
            self.original_catalog_earthquakes = self.process_catalogue(catalog.events)
            self.original_catalog = catalog
            return catalog
        else:
            return None

    def generate_unique_id(self, event_date):
        # Generate a unique ID based on the date and a counter
        unique_id = f"{event_date}_{self.event_counter:02d}"
        self.event_counter += 1  # Increment the counter for the next event
        return unique_id

    def process_catalogue(self, events):
        if not events:
            print("No events to process.")
            return []

        earthquakes = []

        for event in events:
            # Extract event information
            event_id = str(event.resource_id)
            event_time = event.origins[0].time
            event_date = event_time.strftime("%Y-%m-%d")
            event_latitude = event.origins[0].latitude
            event_longitude = event.origins[0].longitude
            event_magnitude = event.magnitudes[0].mag
            event_mag_type = event.magnitudes[0].magnitude_type.lower()
            event_depth = event.origins[0].depth / 1000  # Convert depth to kilometers if needed

            # Generate a unique ID using the new function
            unique_id = self.generate_unique_id(event_date)

            # Create an Earthquake object
            earthquake = Earthquake(
                unique_id=unique_id,
                provider=self.provider,
                event_id=event_id,
                time=event_time.isoformat(),
                lat=event_latitude,
                long=event_longitude,
                mag=event_magnitude,
                mag_type=event_mag_type,
                depth=event_depth,
                epi_distance=None,
                catalogued=True,
                detected=False
            )

            # Update predicted arrivals
            earthquake.update_predicted_arrivals(self.station.latitude, self.station.longitude)

            # Update epicentral distance
            earthquake.update_distance(self.station.latitude, self.station.longitude)

            # Append the earthquake object to the list
            earthquakes.append(earthquake)

        return earthquakes

    def match_and_merge(self, detections, tolerance_p, tolerance_s, p_only=False):

        self.all_day_earthquakes = []

        event_counter = len(self.original_catalog_earthquakes) + 1  # Start counter for the new unique IDs

        for earthquake in self.original_catalog_earthquakes:
            highest_p_confidence = 0
            highest_s_confidence = 0
            best_p_detection = None
            best_s_detection = None

            for detection in detections:
                detected_time = UTCDateTime(detection['peak_time'])
                detected_phase = detection['phase']
                detected_confidence = detection['peak_confidence']

                if detected_phase == 'P' and earthquake.p_predicted and abs(
                        detected_time - UTCDateTime(earthquake.p_predicted)) <= tolerance_p:
                    if detected_confidence > highest_p_confidence:
                        highest_p_confidence = detected_confidence
                        best_p_detection = detection['peak_time']

                if not p_only and detected_phase == 'S' and earthquake.s_predicted and abs(
                        detected_time - UTCDateTime(earthquake.s_predicted)) <= tolerance_s:
                    if detected_confidence > highest_s_confidence:
                        highest_s_confidence = detected_confidence
                        best_s_detection = detection['peak_time']

            # Update earthquake with the best detected times and confidences
            if best_p_detection:
                earthquake.p_detected = best_p_detection
                earthquake.p_confidence = highest_p_confidence
                earthquake.detected = True

            if not p_only and best_s_detection:
                earthquake.s_detected = best_s_detection
                earthquake.s_confidence = highest_s_confidence
                earthquake.detected = True

            # Filter out the matched detections to avoid re-matching
            detections = [d for d in detections if
                          not (d['peak_time'] == best_p_detection or (
                                  not p_only and d['peak_time'] == best_s_detection))]

            # 将处理过的地震对象添加到 all_day_earthquakes
            self.all_day_earthquakes.append(earthquake)

        # Add unmatched detections as new earthquake objects
        for detection in detections:
            unique_id = f"{self.station.report_date.strftime('%Y-%m-%d')}_{event_counter:02d}"
            event_counter += 1

            new_earthquake = Earthquake(
                unique_id=unique_id,
                provider="Detection",
                event_id=None,
                time=detection['peak_time'].isoformat() if isinstance(detection['peak_time'], UTCDateTime) else
                detection['peak_time'],
                lat=None,
                long=None,
                mag=None,
                mag_type=None,
                depth=None,
                epi_distance=None,
                p_predicted=None,
                s_predicted=None,
                p_detected=detection['peak_time'].isoformat() if detection['phase'] == 'P' and isinstance(
                    detection['peak_time'], UTCDateTime) else None,
                s_detected=detection['peak_time'].isoformat() if not p_only and detection[
                    'phase'] == 'S' and isinstance(detection['peak_time'], UTCDateTime) else None,
                p_confidence=detection['peak_confidence'] if detection['phase'] == 'P' else None,
                s_confidence=detection['peak_confidence'] if not p_only and detection['phase'] == 'S' else None,
                catalogued=False,
                detected=True
            )
            self.all_day_earthquakes.append(new_earthquake)

        return self.all_day_earthquakes

    def print_summary(self):
        total_catalogued = len([eq for eq in self.all_day_earthquakes if eq.catalogued])
        detected_catalogued = len([eq for eq in self.all_day_earthquakes if eq.catalogued and eq.detected])
        detected_not_catalogued_count = len(
            [eq for eq in self.all_day_earthquakes if eq.detected and not eq.catalogued])

        return detected_catalogued, detected_not_catalogued_count

    def request_bgs_catalogue(self):
        """
        Construct the BGS catalog request URL based on the station information and search parameters.

        Returns:
        --------
        str
            The URL string to request the earthquake catalog in CSV format from BGS.
        """
        # Calculate start and end time
        starttime = self.station.report_date - 30 * 5  # 5 minutes before midnight on the day before
        endtime = self.station.report_date + (24 * 3600) + 30 * 5  # 5 minutes after midnight on the day after

        # Convert radmax from degrees to kilometers (assuming Earth radius as 6371 km)
        radius_km = self.radmax * 111.32

        max_radius_km = 500.0
        if radius_km > max_radius_km:
            radius_km = max_radius_km

        # Format dates for the URL
        date1 = starttime.strftime('%Y-%m-%d')
        date2 = endtime.strftime('%Y-%m-%d')

        # Construct the BGS URL
        url = (
            f"https://www.earthquakes.bgs.ac.uk/cgi-bin/get_events"
            f"?lat0={self.station.latitude}"
            f"&lon0={self.station.longitude}"
            f"&radius={radius_km:.2f}"
            f"&date1={date1}"
            f"&date2={date2}"
            f"&dep1="
            f"&dep2="
            f"&mag1={self.minmag}"
            f"&mag2={self.maxmag}"
            f"&nsta1="
            f"&nsta2="
            f"&output=csv"
        )

        return url

    def update_summary_csv(self, deployed=True, max_attempts=3):
        date_str = self.station.report_date.strftime('%Y-%m-%d')
        station_network = self.station.network
        station_code = self.station.code
        repo_dir = os.path.join("data", f"{station_network}.{station_code}")

        # 创建临时文件夹
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        headers = ["network", "code", "date", "unique_id", "provider", "event_id", "time", "lat", "long", "mag",
                   "mag_type", "depth", "epi_distance", "p_predicted", "s_predicted", "p_detected", "s_detected",
                   "p_confidence", "s_confidence", "p_error", "s_error", "catalogued", "detected", "plot_path"]

        # 设置 plot_path
        for eq in self.all_day_earthquakes:
            if eq.detected and eq.catalogued:
                if deployed:
                    file_name = f"{eq.unique_id}_event_plot.png"
                    eq.plot_path = f"https://raw.githubusercontent.com/{REPO_NAME}/main/{repo_dir}/{date_str}/report/{file_name}?raw=true"
                else:
                    eq.plot_path = os.path.join(self.station.report_folder, f"{eq.unique_id}_event_plot.png")

        # 创建新的 DataFrame 包含所有的事件
        new_data = pd.DataFrame([{
            "network": station_network,
            "code": station_code,
            "date": date_str,
            "unique_id": eq.unique_id,
            "provider": eq.provider,
            "event_id": eq.event_id,
            "time": eq.time.isoformat(),
            "lat": eq.lat,
            "long": eq.long,
            "mag": eq.mag,
            "mag_type": eq.mag_type,
            "depth": eq.depth,
            "epi_distance": eq.epi_distance,
            "p_predicted": eq.p_predicted.isoformat() if eq.p_predicted else None,
            "s_predicted": eq.s_predicted.isoformat() if eq.s_predicted else None,
            "p_detected": eq.p_detected.isoformat() if eq.p_detected else None,
            "s_detected": eq.s_detected.isoformat() if eq.s_detected else None,
            "p_confidence": eq.p_confidence,
            "s_confidence": eq.s_confidence,
            "p_error": eq.p_error,
            "s_error": eq.s_error,
            "catalogued": eq.catalogued,
            "detected": eq.detected,
            "plot_path": eq.plot_path
        } for eq in self.all_day_earthquakes], columns=headers)

        sha_value = None  # 用于存储上传后的SHA值

        if deployed:
            # 使用 read_summary_csv 读取现有的 summary 数据
            existing_data, status = read_summary_csv(station_network, station_code, deployed=True)

            if status == "loaded":
                print(f"Existing summary file found. Merging with new data.")
                existing_data = existing_data[~existing_data['unique_id'].isin(new_data['unique_id'])]
            elif status == "file_empty":
                print("Existing summary file is empty. Proceeding with new data only.")
                existing_data = pd.DataFrame(columns=headers)
            else:
                print("No existing summary file found or unable to read. Proceeding with new data only.")
                existing_data = pd.DataFrame(columns=headers)

            # 合并新的数据
            full_columns = set(new_data.columns).union(set(existing_data.columns))
            new_data = new_data.reindex(columns=full_columns, fill_value=pd.NA)
            existing_data = existing_data.reindex(columns=full_columns, fill_value=pd.NA)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)

            # 上传合并后的 summary 文件
            temp_summary_path = os.path.join(temp_dir, f"temp_summary_{int(time.time())}.csv")
            updated_data.to_csv(temp_summary_path, index=False)
            print(f"Merged data saved to temporary file '{temp_summary_path}'.")

            # 尝试上传新文件，覆盖 summary
            summary_file_path = f"{repo_dir}/processed_earthquakes_summary.csv"
            backup_summary_file_path = f"{repo_dir}/processed_earthquakes_summary_backup.csv"

            for attempt in range(max_attempts):
                try:
                    print(f"Uploading the new summary file to GitHub as '{summary_file_path}'.")
                    upload_file_to_github(temp_summary_path, summary_file_path)
                    print(f"Summary file '{summary_file_path}' successfully uploaded to GitHub.")

                    # 获取最新文件的SHA值
                    sha_value = get_file_sha(summary_file_path)
                    print(f"SHA of the uploaded summary file: {sha_value}")
                    break
                except GithubException as e:
                    print(f"Attempt {attempt + 1} to upload file failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        print("Retrying...")
                        time.sleep(2)
                    else:
                        print("Max attempts reached. Could not upload the summary file.")
                        return None, None

            # 上传备份文件
            try:
                print(f"Uploading backup summary file to GitHub as '{backup_summary_file_path}'.")
                upload_file_to_github(temp_summary_path, backup_summary_file_path)
                print(f"Backup summary file '{backup_summary_file_path}' successfully uploaded to GitHub.")
            except GithubException as e:
                print(f"Error uploading backup summary file: {str(e)}")

            # 删除临时文件
            os.remove(temp_summary_path)
            print(f"Temporary file '{temp_summary_path}' has been deleted.")

        else:
            # 本地模式处理 summary 文件
            summary_file_path = os.path.join(self.station.station_folder, "processed_earthquakes_summary.csv")
            backup_summary_file_path = os.path.join(self.station.station_folder,
                                                    "processed_earthquakes_summary_backup.csv")

            if os.path.exists(summary_file_path):
                print(f"Existing summary file found locally: '{summary_file_path}'. Merging with new data.")
                existing_data = pd.read_csv(summary_file_path)
                existing_data = existing_data[~existing_data['unique_id'].isin(new_data['unique_id'])]
            else:
                print("No existing summary file found locally. Proceeding with new data only.")
                existing_data = pd.DataFrame(columns=headers)

            full_columns = set(new_data.columns).union(set(existing_data.columns))
            new_data = new_data.reindex(columns=full_columns, fill_value=pd.NA)
            existing_data = existing_data.reindex(columns=full_columns, fill_value=pd.NA)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)

            # 保存 summary 文件
            updated_data.to_csv(summary_file_path, index=False)
            print(f"Processed summary saved to '{summary_file_path}'")

            # 复制一份作为备份文件
            updated_data.to_csv(backup_summary_file_path, index=False)
            print(f"Backup summary saved to '{backup_summary_file_path}'")

        return updated_data
