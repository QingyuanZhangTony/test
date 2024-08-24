import os

import matplotlib.pyplot as plt
import numpy as np

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel

from stream import StreamData


def calculate_time_error(predicted, detected):
    if not predicted or not detected:
        return None  # Return None or NaN for missing values
    try:
        predicted_dt = UTCDateTime(predicted)
        detected_dt = UTCDateTime(detected)

        # Calculate the time difference in seconds
        delta = detected_dt - predicted_dt

        return delta  # Return the numeric difference directly
    except Exception as e:
        return None  # Return None for any errors encountered


def predict_arrivals(lat, long, depth, time, station_latitude, station_longitude):
    model = TauPyModel(model="iasp91")
    distance_deg = gps2dist_azimuth(lat, long, station_latitude, station_longitude)[
                       0] / 1000.0 / 111.195  # Convert meters to degrees

    arrivals = model.get_ray_paths(source_depth_in_km=depth / 1000.0, distance_in_degree=distance_deg,
                                   phase_list=["P", "S"])

    p_arrival, s_arrival = None, None
    for arrival in arrivals:
        if arrival.name == "P":
            p_arrival = time + arrival.time
        elif arrival.name == "S":
            s_arrival = time + arrival.time

    return p_arrival, s_arrival



class Earthquake:
    def __init__(self, unique_id, provider, event_id, time, lat, long, mag, mag_type, depth, epi_distance,
                 p_predicted=None, s_predicted=None, p_detected=None, s_detected=None,
                 p_confidence=None, s_confidence=None, p_error=None, s_error=None, catalogued=True, detected=False):

        self.unique_id = unique_id
        self.provider = provider
        self.event_id = event_id
        self.time = UTCDateTime(time)
        self.lat = float(lat) if lat is not None else None
        self.long = float(long) if long is not None else None
        self.mag = float(mag) if mag is not None else None
        self.mag_type = mag_type
        self.depth = float(depth) if depth is not None else None
        self.epi_distance = float(epi_distance) if epi_distance is not None else None
        self.p_predicted = UTCDateTime(p_predicted) if p_predicted else None
        self.s_predicted = UTCDateTime(s_predicted) if s_predicted else None
        self.p_detected = UTCDateTime(p_detected) if p_detected else None
        self.s_detected = UTCDateTime(s_detected) if s_detected else None
        self.p_confidence = float(p_confidence) if p_confidence is not None else None
        self.s_confidence = float(s_confidence) if s_confidence is not None else None
        self.p_error = float(p_error) if p_error is not None else None
        self.s_error = float(s_error) if s_error is not None else None
        self.catalogued = catalogued
        self.detected = detected
        self.plot_path = None

        self.event_stream = StreamData(self)  # Create a StreamData object for this earthquake

    def __str__(self):
        return (f"Earthquake ID: {self.unique_id}\n"
                f"Provider: {self.provider}\n"
                f"Provider Event ID: {self.event_id}\n"
                f"Time: {self.time}\n"
                f"Latitude: {self.lat}, Longitude: {self.long}\n"
                f"Depth: {self.depth} km\n"
                f"Magnitude: {self.mag} {self.mag_type}\n"
                f"Epicentral Distance: {self.epi_distance} km\n"
                f"P Predicted: {self.p_predicted}, S Predicted: {self.s_predicted}\n"
                f"P Detected: {self.p_detected}, S Detected: {self.s_detected}\n"
                f"P Confidence: {self.p_confidence}, S Confidence: {self.s_confidence}\n"
                f"P Error: {self.p_error}, S Error: {self.s_error}\n"
                f"Catalogued: {'Yes' if self.catalogued else 'No'}, Detected: {'Yes' if self.detected else 'No'}\n"
                "----------------------------------------")

    def download_event_stream(self, station, duration=60):
        channel = "*Z*"
        location = "*"
        p_predicted_time = UTCDateTime(self.p_predicted)
        date_str = p_predicted_time.strftime("%Y-%m-%d")
        path = station.monitoring_folder

        nslc = f"{station.network}.{station.code}.{location}.{channel}".replace("*", "")
        filename = f"{date_str}_{nslc}_{self.unique_id}.mseed"
        filepath = os.path.join(path, filename)

        client = Client(station.url)
        download_duration = 600  # Fixed download duration
        start_time = p_predicted_time - download_duration / 2
        end_time = p_predicted_time + download_duration / 2

        try:
            event_stream = client.get_waveforms(station.network, station.code, location, channel, start_time, end_time,
                                                attach_response=True)
            event_stream.merge()
            for tr in event_stream:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled(fill_value=0)
            os.makedirs(path, exist_ok=True)
            event_stream.write(filepath)
            print(f"Event data for {self.unique_id} successfully downloaded.")

            # Slice the stream to the desired duration
            slice_start_time = p_predicted_time - duration / 2
            slice_end_time = p_predicted_time + duration / 2
            sliced_stream = event_stream.slice(starttime=slice_start_time, endtime=slice_end_time)

            self.event_stream.original_stream = sliced_stream  # Set the sliced stream object in the earthquake object
            return sliced_stream
        except Exception as e:
            print(f"Failed to download event data: {e}")
            return None

    def update_errors(self):
        self.p_error = calculate_time_error(self.p_predicted, self.p_detected)
        self.s_error = calculate_time_error(self.s_predicted, self.s_detected)

    def update_distance(self, station_lat, station_lon):
        if self.lat is not None and self.long is not None:
            self.epi_distance = locations2degrees(station_lat, station_lon, self.lat,
                                                  self.long) * 111.195  # Convert to km
        else:
            self.epi_distance = None

    def update_predicted_arrivals(self, station_latitude, station_longitude):
        p_arrival, s_arrival = predict_arrivals(self.lat, self.long, self.depth, self.time, station_latitude,
                                                station_longitude)
        self.p_predicted = p_arrival
        self.s_predicted = s_arrival

