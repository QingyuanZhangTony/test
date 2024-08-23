import os
import time
import numpy as np
import streamlit as st
from obspy import UTCDateTime, read, Stream
from obspy.clients.fdsn import Client

from stream import StreamData


class Station:
    """
    A class representing a seismic station, responsible for managing its data and
    generating paths for data storage. It also includes methods for downloading
    seismic data and fetching station coordinates.

    Attributes:
    -----------
    network : str
        The network code of the station.
    code : str
        The station code.
    url : str
        The URL of the data provider.
    report_date : UTCDateTime
        The date for which the report is generated.
    latitude : float or None
        The latitude of the station.
    longitude : float or None
        The longitude of the station.
    station_folder : str or None
        The path to the station's data folder.
    date_folder : str or None
        The path to the specific date's data folder.
    report_folder : str or None
        The path to the report folder for the specified date.
    stream : StreamData
        An instance of StreamData for managing seismic data streams.
    """

    def __init__(self, network, code, url, report_date=None, latitude=None, longitude=None):
        """
        Initialize the Station instance.

        Parameters:
        -----------
        network : str
            The network code of the station.
        code : str
            The station code.
        url : str
            The URL of the data provider.
        report_date : str or UTCDateTime, optional
            The date for which the report is generated. Defaults to the current date.
        latitude : float, optional
            The latitude of the station.
        longitude : float, optional
            The longitude of the station.
        """
        self.network = network
        self.code = code
        self.url = url
        self.report_date = UTCDateTime(report_date) if report_date else UTCDateTime()

        self.latitude = latitude
        self.longitude = longitude
        self.station_folder = None
        self.date_folder = None
        self.report_folder = None

        self.stream = StreamData(self)

        self.generate_path(self.report_date)

    def __str__(self):
        """
        Return a string representation of the Station instance.

        Returns:
        --------
        str:
            A string describing the station's details, including location, report date,
            and folder paths.
        """
        return (f"Station {self.network}.{self.code} at {self.url}\n"
                f"Location: {self.latitude}, {self.longitude}\n"
                f"Report Date: {self.report_date.strftime('%Y-%m-%d')}\n"
                f"Data Folder: {self.date_folder}\n"
                f"Report Folder: {self.report_folder}\n")

    @staticmethod
    def fetch_coordinates(network, station_code, url, retries=1):
        """
        Fetch the coordinates of the station from the data provider.

        This method attempts to retrieve the latitude and longitude of the station
        from the specified data provider URL. If a 502 Bad Gateway error occurs,
        the method retries the request a specified number of times.

        Parameters:
        -----------
        network : str
            The network code.
        station_code : str
            The station code.
        url : str
            The data provider URL.
        retries : int, optional
            Number of retry attempts for HTTP 502 errors. Defaults to 1.

        Returns:
        --------
        tuple:
            A tuple containing the latitude and longitude of the station if successful,
            else (None, None).
        """
        retry_count = retries
        while retry_count >= 0:
            try:
                client = Client(url)
                endtime = UTCDateTime()
                inventory = client.get_stations(network=network, station=station_code, endtime=endtime, level='station')
                latitude = inventory[0][0].latitude
                longitude = inventory[0][0].longitude
                return latitude, longitude
            except Exception as e:
                if '502' in str(e):
                    st.warning(f"HTTP 502 error encountered: Retrying after 15 seconds...")
                    time.sleep(15)
                    retry_count -= 1
                else:
                    st.error(f"Error fetching station coordinates: {e}")
                    break

        st.error("Failed to fetch coordinates after retrying. Error: HTTP 502 Bad Gateway")
        return None, None

    def generate_path(self, date):
        """
        Generate the directory paths for storing station data and reports.

        This method constructs the directory paths based on the station's network and
        code, as well as the specified date. The directories are created if they do
        not already exist.

        Parameters:
        -----------
        date : UTCDateTime or str
            The date for which the paths are generated. It can be either a UTCDateTime
            object or a string in 'YYYY-MM-DD' format.

        Raises:
        -------
        ValueError:
            If the date is not a UTCDateTime object or a valid string format.
        """
        if isinstance(date, UTCDateTime):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            date_str = date
        else:
            raise ValueError("Date must be a UTCDateTime object or a string in 'YYYY-MM-DD' format")

        # Construct the directory paths
        base_dir = os.getcwd()
        self.station_folder = os.path.join(base_dir, "data", f"{self.network}.{self.code}")
        self.date_folder = os.path.join(self.station_folder, date_str)
        self.report_folder = os.path.join(self.date_folder, "report")

        # Ensure the directories exist
        os.makedirs(self.station_folder, exist_ok=True)
        os.makedirs(self.date_folder, exist_ok=True)
        os.makedirs(self.report_folder, exist_ok=True)

    def download_stream_part(self, start_time, end_time):
        """
        下载特定时间段的地震数据流。

        参数:
        ----------
        start_time : UTCDateTime
            起始时间。
        end_time : UTCDateTime
            结束时间。

        返回:
        ----------
        Stream
            下载的数据流。
        """
        client = Client(self.url)
        partial_st = client.get_waveforms(self.network, self.code, "*", "*Z*", start_time, end_time, attach_response=True)
        partial_st.merge(method=0)
        for tr in partial_st:
            if isinstance(tr.data, np.ma.masked_array):
                tr.data = tr.data.filled(fill_value=0)  # 将掩码值填充为0
        return partial_st

    def check_existing_stream(self, overwrite):
        """
        Check if the stream data file already exists and whether it should be overwritten.

        Parameters:
        -----------
        overwrite : bool
            Whether to overwrite the existing file.

        Returns:
        --------
        bool
            True if the file exists and should not be overwritten, False otherwise.
        """
        nslc = f"{self.network}.{self.code}.*.*Z*".replace("*", "")
        filename = f"{self.report_date.strftime('%Y-%m-%d')}_{nslc}.mseed"
        filepath = os.path.join(self.date_folder, filename)

        return os.path.isfile(filepath) and not overwrite

    def merge_streams(self, parts):
        """
        合并给定的多个数据流部分，并将合并后的流存储在 original_stream 中。

        Parameters:
        -----------
        parts : list of Stream
            需要合并的数据流部分。

        Returns:
        --------
        bool
            如果合并成功，返回 True；否则返回 False。
        """
        try:
            full_stream = Stream()
            for part in parts:
                full_stream += part

            full_stream.merge(method=0)  # Final merge before writing
            for tr in full_stream:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled(fill_value=0)

            self.stream.original_stream = full_stream
            return True
        except Exception as e:
            print(f"Error during stream merging: {str(e)}")
            return False
