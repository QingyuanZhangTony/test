import base64
import io
import os
import smtplib
import urllib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import urllib.parse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import weasyprint
from bs4 import BeautifulSoup
from matplotlib.legend_handler import HandlerLine2D
from obspy import read, UTCDateTime
from email.mime.text import MIMEText
import requests
import os

import streamlit as st
import datetime

from github_file import upload_file_to_github, REPO_NAME

import streamlit as st

EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
SMTP_SERVER = st.secrets["SMTP_SERVER"]
SMTP_PORT = st.secrets["SMTP_PORT"]


class Report:
    """
    Base class for generating reports.
    Provides methods for generating HTML, converting images to Base64, and generating PDFs.
    """

    @staticmethod
    def convert_images_to_base64(html_content):
        """
        Convert all image sources in the provided HTML content to Base64 encoded strings.

        Args:
        - html_content (str): The HTML content with potential image tags.

        Returns:
        - str: The modified HTML content with all images converted to Base64.
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Iterate over all img tags and convert images to Base64
        for img_tag in soup.find_all('img'):
            img_src = img_tag['src']

            if img_src.startswith('file:///'):
                # Convert file:/// URL to a local file path
                img_src = urllib.parse.urlparse(img_src).path
                img_src = os.path.normpath(img_src.lstrip('/'))  # Remove leading slash and normalize the path

            if not img_src.startswith('data:'):  # Only convert if it's not already in Base64
                try:
                    with open(img_src, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        img_tag['src'] = f"data:image/png;base64,{img_base64}"
                except FileNotFoundError:
                    print(f"Image file not found: {img_src}")
                except OSError as e:
                    print(f"Error opening file {img_src}: {e}")

        # Convert the soup object back to a string
        return str(soup)

    @staticmethod
    def generate_pdf_buffer_from_html(html_content):
        """
        Generate a PDF report from the provided HTML content.

        Args:
        - html_content (str): The HTML content to be converted into a PDF.

        Returns:
        - bytes: The generated PDF as a byte string.
        """
        pdf_buffer = io.BytesIO()
        weasyprint.HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer

    @staticmethod
    def send_report_via_email(recipient, report_buffer, report_date):
        """
        Send the generated report PDF via email as an attachment.

        Args:
        - recipient (str): The email address of the recipient.
        - report_buffer (BytesIO): The PDF buffer containing the report.
        - report_date (str): The date of the report to be included in the file name.

        Returns:
        - bool: True if the email was sent successfully, False otherwise.
        """
        try:
            # Create the email message
            msg = MIMEMultipart('related')
            msg['Subject'] = "Daily Report"
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = recipient

            # Add a plain text body to the email
            body_text = f"This is an automatic email sent from your app. The Daily Report for {report_date} is attached."
            msg.attach(MIMEText(body_text, 'plain'))

            # Construct the file name using the report date
            filename = f"daily_report_{report_date}.pdf"

            # Attach the PDF file from the buffer
            part = MIMEApplication(report_buffer.getvalue(), Name=filename)
            part['Content-Disposition'] = f'attachment; filename="{filename}"'
            msg.attach(part)

            # Connect to SMTP server
            smtp_obj = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            smtp_obj.starttls()
            smtp_obj.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

            # Send the email
            smtp_obj.sendmail(EMAIL_ADDRESS, recipient, msg.as_string())

            # Disconnect from SMTP server
            smtp_obj.quit()
            print("Email sent successfully.")
            return True

        except Exception as e:
            print(f"Failed to send email: {e}")
            return False


class DailyReport(Report):
    def __init__(self, df, date_str, station_lat, station_lon, fill_map=False, simplified=False, p_only=False):
        """
        Initialize the DailyReport class with the provided parameters.

        Args:
        - df (pd.DataFrame): DataFrame containing earthquake data.
        - date_str (str or datetime.date): The date for the report.
        - station_lat (float): Latitude of the station.
        - station_lon (float): Longitude of the station.
        - fill_map (bool): Whether to fill the map with background imagery.
        """
        self.df = df
        self.date_str = date_str.strftime('%Y-%m-%d') if isinstance(date_str, datetime.date) else str(date_str)
        self.df_filtered = self.df[self.df['date'] == self.date_str]
        self.station_lat = float(station_lat)
        self.station_lon = float(station_lon)
        self.fill_map = fill_map
        self.simplified = simplified
        self.p_only = p_only
        self.report_folder = self.construct_report_folder_path()

    def construct_report_folder_path(self):
        """
        Construct the path to the report folder based on the station information.

        Returns:
        - str: The path to the report folder.
        """
        network = self.df.iloc[0]['network']
        code = self.df.iloc[0]['code']
        base_dir = os.getcwd()
        date_str = self.date_str
        report_folder = os.path.join(base_dir, 'data', f'{network}.{code}', date_str, 'report')
        os.makedirs(report_folder, exist_ok=True)
        return report_folder

    def plot_catalogue(self, deployed=True):
        """
        Plot a world map with all catalogued events and the station location.

        Parameters:
        -----------
        deployed : bool, optional
            If True, uploads the plot to GitHub and returns the URL, otherwise returns the local file path.

        Returns:
        --------
        str:
            The path to the saved plot image or the GitHub URL if deployed.
        """
        earthquakes = self.df_filtered[self.df_filtered['catalogued'] == True]
        title = f"Catalogue Events on {self.date_str}"
        file_name = f'catalogued_plot_{self.date_str}.png'
        output_path = os.path.join(self.report_folder, file_name)

        # Extract station-related information from df
        network = self.df.iloc[0]['network']
        code = self.df.iloc[0]['code']
        latitude = self.station_lat
        longitude = self.station_lon

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=longitude)})
        ax.set_global()
        ax.coastlines()

        # Set map style based on fill_map option
        if self.fill_map:
            ax.stock_img()
            cmap = plt.get_cmap('autumn')
            station_color = '#7F27FF'
            marker_color = '#FAA300'
        else:
            cmap = plt.get_cmap('viridis')
            station_color = '#F97300'
            marker_color = '#135D66'

        # Plot station location
        ax.plot(longitude, latitude, marker='^', color=station_color, markersize=16, linestyle='None',
                transform=ccrs.Geodetic(), label=f'Station {code}')

        norm = plt.Normalize(1, 10)

        # Plot earthquakes
        detected_count = 0
        undetected_count = 0
        for _, eq in earthquakes.iterrows():
            color = cmap(norm(eq['mag']))
            if eq['detected']:
                marker = 's'  # Square for detected
                detected_count += 1
            else:
                marker = 'o'  # Circle for undetected
                undetected_count += 1
            ax.plot(eq['long'], eq['lat'], marker=marker, color=color, markersize=10, markeredgecolor='white',
                    transform=ccrs.Geodetic())

        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=32.5, fraction=0.015, shrink=0.9)
        cbar.set_label('Magnitude')

        # Add title and legend
        plt.title(title, fontsize=15)

        detected_marker = plt.Line2D([], [], color=marker_color, marker='s', markersize=10, linestyle='None',
                                     markeredgecolor='white')
        undetected_marker = plt.Line2D([], [], color=marker_color, marker='o', markersize=10, linestyle='None',
                                       markeredgecolor='white')
        station_marker = plt.Line2D([], [], color=station_color, marker='^', markersize=10, linestyle='None',
                                    markeredgecolor='white')

        plt.legend([detected_marker, undetected_marker, station_marker],
                   [f'Detected Earthquake: {detected_count}', f'Undetected Earthquake: {undetected_count}',
                    f'Station {code}'],
                   loc='lower center', handler_map={plt.Line2D: HandlerLine2D(numpoints=1)}, ncol=3)

        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Save the plot locally
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if deployed:
            # Define the GitHub path and upload the file
            repo_file_path = os.path.join('data', f'{network}.{code}', self.date_str, 'report', file_name)
            repo_file_path = repo_file_path.replace("\\", "/")  # Ensure GitHub-compatible path
            upload_file_to_github(output_path, repo_file_path)
            print(f"Uploaded to GitHub: {repo_file_path}")
            return f"https://raw.githubusercontent.com/{REPO_NAME}/main/{repo_file_path}?raw=true"

        return output_path

    def daily_report_header_html(self):
        """
        Generate the header section of the daily earthquake report.

        Returns:
        - str: The HTML string for the daily report header.
        """
        # Find the row corresponding to the given date
        date_row = self.df_filtered.iloc[0]


        network = date_row['network']
        code = date_row['code']
        provider = date_row['provider']

        simplified_status = "Yes" if self.simplified else "No"
        p_only_status = "Yes" if self.p_only else "No"

        header_html = f"""
        <h2>Earthquake Report for {self.date_str}</h2>
        <p>
            <b>Station:</b> <span class="normal" style="display: inline-block; margin-right: 20px;">{network}.{code}</span>
            <b>Catalog Provider:</b> <span class="normal" style="display: inline-block; margin-right: 20px;">{provider}</span>
            <b>Simplified:</b> <span class="normal" style="display: inline-block; margin-right: 20px;">{simplified_status}</span>
            <b>P Wave Only:</b> <span class="normal" style="display: inline-block;">{p_only_status}</span>
        </p>
        """
        return header_html

    def daily_report_catalog_html(self, simplified=True, deployed=True):
        """
        Generate the catalog section of the daily report, including a table of earthquakes.

        Args:
        - simplified (bool): Whether to display only earthquakes that are both catalogued and detected. Defaults to True.
        - deployed (bool): If True, the image path will be the GitHub URL; otherwise, it will be a local file path.

        Returns:
        - str: The HTML string for the catalog section of the report.
        """

        # 构建 image_path，根据部署模式选择本地路径或 GitHub URL
        if deployed:
            network = self.df_filtered.iloc[0]['network']
            code = self.df_filtered.iloc[0]['code']
            file_name = f'catalogued_plot_{self.date_str}.png'
            repo_file_path = os.path.join('data', f'{network}.{code}', self.date_str, 'report', file_name)
            repo_file_path = repo_file_path.replace("\\", "/")
            image_path = f"https://raw.githubusercontent.com/{REPO_NAME}/main/{repo_file_path}?raw=true"
        else:
            image_path = os.path.join(self.report_folder, f'catalogued_plot_{self.date_str}.png')

        # 根据 simplified 标志过滤 DataFrame
        if simplified:
            filtered_df = self.df_filtered[
                (self.df_filtered['detected'] == True) & (self.df_filtered['catalogued'] == True)]
        else:
            filtered_df = self.df_filtered[self.df_filtered['catalogued'] == True]

        # 构建 catalog 部分的 HTML
        catalog_html = f"""
        <h3>Catalogued Earthquake Plot</h3>
        <img src="{image_path}" alt="Catalogued Earthquake Plot for {self.date_str}" />
        <h4>Detected and Catalogued Earthquakes</h4>
        <table>
            <tr>
                <th>Time</th>
                <th>Location</th>
                <th>Magnitude</th>
            </tr>
        """

        # 遍历 filtered DataFrame 行，添加每个地震事件的数据到表格中
        for _, row in filtered_df.iterrows():
            time_str = pd.to_datetime(row['time']).strftime('%Y-%m-%d %H:%M:%S')
            location_str = f"{row['lat']}, {row['long']}"
            magnitude_str = f"{row['mag']} {row['mag_type']}"

            # 如果 simplified 为 False，则将 detected 的地震事件以绿色显示
            if not simplified and row['detected']:
                row_html = f"""
                <tr style="color: green;">
                    <td>{time_str}</td>
                    <td>{location_str}</td>
                    <td>{magnitude_str}</td>
                </tr>
                """
            else:
                row_html = f"""
                <tr>
                    <td>{time_str}</td>
                    <td>{location_str}</td>
                    <td>{magnitude_str}</td>
                </tr>
                """

            catalog_html += row_html

        catalog_html += "</table>"

        return catalog_html

    def daily_report_event_details_html(self):
        """
        Generate the event details section of the daily report for a specific date.

        Returns:
        - str: The HTML string for the event details section of the report.
        """
        # 过滤已检测并且已在目录中的事件
        filtered_df = self.df_filtered[
            (self.df_filtered['catalogued'] == True) & (self.df_filtered['detected'] == True)
            ]

        all_event_details_html = ""

        for _, row in filtered_df.iterrows():
            # Add page break before each event detail
            all_event_details_html += f'<div class="page-break">{EventReport.event_details_html(row, self.simplified, self.p_only)}</div>'

        return all_event_details_html

    def assemble_daily_report_html(self):
        """
        Generate the HTML content for the daily report by combining the header, catalog, and event details sections.

        This method constructs the full HTML content for a daily report by integrating the HTML segments generated
        by `daily_report_header_html`, `daily_report_catalog_html`, and `daily_report_event_details_html`. The
        generated HTML includes styling for various elements such as headers, paragraphs, images, and tables.

        Returns:
        --------
        str:
            The full HTML content as a string, ready for rendering or saving as an HTML file.
        """
        header_html = self.daily_report_header_html()
        catalog_html = self.daily_report_catalog_html(self.simplified)
        event_details_html = self.daily_report_event_details_html()

        full_html_content = f"""
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

                body {{
                    font-family: 'Roboto', sans-serif;
                    margin: 20px;
                    padding: 0;
                }}
                h2 {{
                    text-align: left;
                    font-size: 20px;
                    margin-bottom: 10px;
                }}
                h3 {{
                    font-size: 16px;
                    margin-bottom: 10px;
                }}
                p {{
                    margin: 5px 0;
                    display: inline-block;
                    font-weight: bold;
                }}
                span.normal {{
                    font-weight: normal;
                    margin-right: 20px;
                }}
                img {{
                    width: 100%;
                    height: auto;
                    margin: 20px 0;
                }}
                .earthquake-image {{
                    width: 110%;
                    height: auto;
                    display: block;
                    margin: 20px 0;
                    transform: translateX(-5%);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                    font-size: 12px;
                }}
                table, th, td {{
                    border: 1px solid black;
                }}
                th, td {{
                    padding: 5px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .col1, .col3 {{
                    width: 20%;
                }}
                .col2, .col4 {{
                    width: 30%;
                }}
                .page-break {{
                    page-break-before: always;
                }}
            </style>
        </head>
        <body>
            {header_html}
            {catalog_html}
            {event_details_html}
        </body>
        </html>
        """

        return full_html_content

    def export_daily_report_pdf(self, html_content, deployed=True):
        """
        Generate a PDF report from the provided HTML content and save it or upload it to GitHub.

        This method converts the provided HTML content into a PDF file. Depending on the `deployed`
        flag, it either saves the PDF locally or uploads it to the specified GitHub repository.

        Parameters:
        -----------
        html_content : str
            The HTML content to be converted into a PDF.
        deployed : bool, optional
            If True, upload the PDF to GitHub; otherwise, save it locally. Defaults to True.

        Returns:
        --------
        str:
            The path to the saved PDF file or the GitHub URL if deployed.
        """
        file_name = f'daily_report_{self.date_str}.pdf'
        report_path = os.path.join(self.report_folder, file_name)

        # Generate the PDF buffer
        pdf_buffer = self.generate_pdf_buffer_from_html(html_content)

        if deployed:
            # If deployed, upload the PDF to GitHub
            repo_dir = os.path.join("data", f"{self.df_filtered.iloc[0]['network']}.{self.df_filtered.iloc[0]['code']}",
                                    self.date_str,
                                    'report')
            repo_file_path = os.path.join(repo_dir, file_name).replace("\\", "/")

            # Save the PDF to a temporary local file before uploading
            with open(report_path, 'wb') as f:
                f.write(pdf_buffer.read())

            # Upload the PDF to GitHub
            upload_file_to_github(report_path, repo_file_path)
            print(f"PDF uploaded to GitHub: {repo_file_path}")

            # Return the GitHub URL
            return f"https://raw.githubusercontent.com/{REPO_NAME}/main/{repo_file_path}?raw=true"

        else:
            # If not deployed, save the PDF locally
            with open(report_path, 'wb') as f:
                f.write(pdf_buffer.read())

            print(f"PDF generated and saved as {report_path}")
            return report_path

    def save_html_to_file(self, html_content):
        """
        Save the generated HTML content to a file in the appropriate directory.

        This method writes the provided HTML content to a file, saving it with a name based on the
        current date in the specified report folder. This file can then be used for further reference
        or as an input for other processes.

        Parameters:
        -----------
        html_content : str
            The HTML content to be saved as a file.

        Returns:
        --------
        str:
            The path to the saved HTML file.
        """
        html_filename = f'daily_report_{self.date_str}.html'
        html_file_path = os.path.join(self.report_folder, html_filename)

        with open(html_file_path, 'w') as f:
            f.write(html_content)

        return html_file_path


class EventReport(Report):
    """
    Class for generating event reports.

    This class is responsible for creating detailed reports for seismic events. It
    inherits from the `Report` base class and provides additional methods for
    generating HTML content and visualizations specific to earthquake events.
    """

    def __init__(self, df_row):
        """
        Initialize an EventReport instance with a specific row of data.

        Parameters:
        -----------
        df_row : pd.Series
            The row of the DataFrame containing the earthquake details.
        """
        self.df_row = df_row

    @staticmethod
    def event_details_html(df_row, simplified, p_only, deployed=True):
        """
        Generate the complete event report including the header and earthquake details.

        This method creates an HTML string that contains the header information and detailed
        data about the earthquake event, such as the station details, event ID, and wave
        arrival times. The HTML also includes an image plot associated with the event.

        Parameters:
        -----------
        df_row : pd.Series
            The row of the DataFrame containing the earthquake details.
        simplified : bool, optional
            If True, hides some lines depending on the p_only flag.
        p_only : bool, optional
            If True, hides S wave-related details.
        deployed : bool, optional
            If True, expects the image to be a URL; otherwise, treats it as a local file path.

        Returns:
        --------
        str:
            The complete HTML string for the event report.
        """
        # Generate the header section
        report_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        network = df_row['network']
        code = df_row['code']

        header_html = f"""
        <h3>Event Report For Earthquake {df_row['unique_id']}</h3>
        <p><b>Station:</b> <span class="normal">{network}.{code}</span> 
           <b>Issued At:</b> <span class="normal">{report_time}</span></p>
        """

        # Generate the event details section
        plot_path = df_row['plot_path'].replace("\\", "/")  # 确保URL格式的路径正确

        # Convert the time to a more readable format
        readable_time = pd.to_datetime(df_row['time']).strftime('%Y-%m-%d %H:%M:%S')

        # Format the epicentral distance to two decimal places
        formatted_distance = f"{df_row['epi_distance']:.2f}" if pd.notnull(df_row['epi_distance']) else 'N/A'

        earthquake_info_html = f"""
        <div style="page-break-before: always;"> <!-- Ensures a page break before each event -->
            {header_html}
        """

        if deployed:
            # 部署模式，直接使用URL
            earthquake_info_html += f"""
            <img src="{plot_path}" alt="Event Plot" class="earthquake-image"/>
            """
        else:
            # 本地模式，将路径转换为合适的本地文件路径
            local_plot_path = f"file:///{plot_path}"
            earthquake_info_html += f"""
            <img src="{local_plot_path}" alt="Event Plot" class="earthquake-image"/>
            """

        earthquake_info_html += f"""
            <table>
                <tr><th class="col1">Time:</th><td class="col2">{readable_time}</td>
                    <th class="col3">Event ID:</th><td class="col4">{df_row['unique_id']}</td></tr>
                <tr><th class="col1">Latitude, Longitude:</th><td class="col2">{df_row['lat']}, {df_row['long']}</td>
                    <th class="col3">Epicentral Distance:</th><td class="col4">{formatted_distance} km</td></tr>
        """

        # 如果 simplified 为 False，添加 Provider 和 Resource ID 行
        if not simplified:
            earthquake_info_html += f"""
                <tr><th class="col1">Provider:</th><td class="col2">{df_row['provider']}</td>
                    <th class="col3">Resource ID:</th><td class="col4">{df_row['event_id']}</td></tr>
            """

        earthquake_info_html += f"""
                <tr><th class="col1">Depth:</th><td class="col2">{df_row['depth']} km</td>
                    <th class="col3">Magnitude:</th><td class="col4">{df_row['mag']} {df_row['mag_type']}</td></tr>
        """

        # 添加 P-wave detected time 和 P-wave error
        p_predicted_time = pd.to_datetime(df_row['p_predicted']).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(
            df_row['p_predicted']) else 'N/A'
        p_detected_time = pd.to_datetime(df_row['p_detected']).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(
            df_row['p_detected']) else 'N/A'

        earthquake_info_html += f"""
                <tr><th class="col1">P Predicted Time:</th><td class="col2">{p_predicted_time}</td>
                    <th class="col3">P Detected Time:</th><td class="col4">{p_detected_time}</td></tr>
        """

        # 如果 p_only 为 False，添加 S-wave detected time 和 S-wave error
        if not p_only:
            s_predicted_time = pd.to_datetime(df_row['s_predicted']).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(
                df_row['s_predicted']) else 'N/A'
            s_detected_time = pd.to_datetime(df_row['s_detected']).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(
                df_row['s_detected']) else 'N/A'

            earthquake_info_html += f"""
                <tr><th class="col1">S Predicted Time:</th><td class="col2">{s_predicted_time}</td>
                    <th class="col3">S Detected Time:</th><td class="col4">{s_detected_time}</td></tr>
            """

        # 如果 simplified 为 False，添加 error 和 confidence 细节
        if not simplified:
            earthquake_info_html += f"""
                <tr><th class="col1">P Time Error:</th><td class="col2">{df_row['p_error'] or 'N/A'}</td>
                    <th class="col3">P Confidence:</th><td class="col4">{df_row['p_confidence'] or 'N/A'}</td></tr>
            """
            if not p_only:
                earthquake_info_html += f"""
                    <tr><th class="col1">S Time Error:</th><td class="col2">{df_row['s_error'] or 'N/A'}</td>
                        <th class="col3">S Confidence:</th><td class="col4">{df_row['s_confidence'] or 'N/A'}</td></tr>
                """

        earthquake_info_html += "</table></div>"

        return earthquake_info_html

    def assemble_event_report_html(self, simplified, p_only):
        """
        Assemble the complete HTML content for the report.

        This method combines the different sections of the event report into a
        single HTML document that is ready for rendering or saving.

        Returns:
        --------
        str:
            The complete HTML string for the report.
        """
        # Generate the different sections of the report
        event_full_html = self.event_details_html(self.df_row, simplified, p_only)

        # Combine all parts into a single HTML document
        full_html = f"""
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

                body {{
                    font-family: 'Roboto', sans-serif;
                    margin: 20px;
                    padding: 0;
                    line-height: 1.5;  
                }}
                h2 {{
                    text-align: left;
                    font-size: 20px;
                    margin-bottom: 5px;  
                }}
                p {{
                    margin: 2px 0;  
                    display: inline-block;
                    font-weight: bold;
                }}
                span.normal {{
                    font-weight: normal;
                    margin-right: 20px;
                }}
                img {{
                    width: 110%;
                    height: auto;
                    display: block;
                    margin: 20px 0;
                    transform: translateX(-5%);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 5px 0;
                    font-size: 12px;
                }}
                table, th, td {{
                    border: 1px solid black;
                }}
                th, td {{
                    padding: 5px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .col1, .col3 {{
                    width: 20%;
                }}
                .col2, .col4 {{
                    width: 30%;
                }}
                .page-break {{
                    page-break-before: always;
                }}
            </style>
        </head>
        <body>
            {event_full_html}
        </body>
        </html>
        """
        return full_html

    def plot_waveform(self, starttime, endtime, ax=None, p_only=False):
        """
        Plot the normalized waveform around the detected and predicted P-wave and S-wave times.

        This method slices the seismic data stream between the given start and end times,
        then plots the normalized waveform. If specified, it highlights the detected and
        predicted arrival times for P-waves and S-waves.

        Parameters:
        -----------
        starttime : UTCDateTime
            The start time for slicing the stream.
        endtime : UTCDateTime
            The end time for slicing the stream.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axes to plot on. If None, a new figure is created.
        p_only : bool, optional
            If True, only plot P-wave times; otherwise, plot both P-wave and S-wave times.

        Returns:
        --------
        matplotlib.axes._subplots.AxesSubplot:
            The axis with the plotted waveform.
        """
        df_row = self.df_row
        try:
            date_str = df_row['date']
            network = df_row['network']
            code = df_row['code']

            base_dir = os.getcwd()
            stream_file_name = f"{date_str}_{network}.{code}..Z.processed.mseed"
            stream_path = os.path.join(base_dir, 'data', f'{network}.{code}', date_str, stream_file_name)

            stream = read(stream_path)
            trace = stream.slice(starttime=starttime, endtime=endtime)

            if len(trace) == 0:
                print("No data in the trace.")
                return None

            start_time = trace[0].stats.starttime
            end_time = trace[0].stats.endtime

            if ax is None:
                fig, ax = plt.subplots(figsize=(13, 4))

            ax.plot(trace[0].times(), trace[0].data / np.amax(np.abs(trace[0].data)), 'k', label=trace[0].stats.channel)
            ax.set_ylabel('Normalized Amplitude')

            # Plot P-wave detected and predicted times
            for t, color, style, label in [(df_row['p_detected'], "C0", "-", "Detected P Arrival"),
                                           (df_row['p_predicted'], "C0", "--", "Predicted P Arrival")]:
                if pd.notna(t):
                    t_utc = UTCDateTime(t)
                    ax.axvline(x=t_utc - start_time, color=color, linestyle=style, label=label, linewidth=0.8)

            if not p_only:
                # Plot S-wave detected and predicted times
                for t, color, style, label in [(df_row['s_detected'], "C1", "-", "Detected S Arrival"),
                                               (df_row['s_predicted'], "C1", "--", "Predicted S Arrival")]:
                    if pd.notna(t):
                        t_utc = UTCDateTime(t)
                        ax.axvline(x=t_utc - start_time, color=color, linestyle=style, label=label, linewidth=0.8)

            ax.set_xlim(0, end_time - start_time)

            num_ticks = 5
            x_ticks = np.linspace(0, end_time - start_time, num_ticks)
            x_labels = [(start_time + t).strftime('%H:%M:%S') for t in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=0)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right')

            return ax

        except Exception as e:
            print(f"Error plotting waveform: {e}")
            return None

    def plot_prediction_confidence(self, starttime, endtime, ax=None, p_only=False):
        """
        Plot the prediction confidence for P and S waves from the annotated stream.

        This method generates a plot of the prediction confidence levels for P-waves and S-waves
        over a specified time range. The plot is derived from the annotated seismic data stream.

        Parameters:
        -----------
        starttime : UTCDateTime
            The start time for slicing the stream.
        endtime : UTCDateTime
            The end time for slicing the stream.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axes to plot on. If None, a new figure is created.
        p_only : bool, optional
            If True, only plot P-wave and Detection; otherwise, plot both P-wave and S-wave.

        Returns:
        --------
        matplotlib.axes._subplots.AxesSubplot:
            The axis with the plotted prediction confidence.
        """
        df_row = self.df_row
        try:
            date_str = df_row['date']
            network = df_row['network']
            code = df_row['code']

            base_dir = os.getcwd()
            annotated_stream_file_name = f"{date_str}_{network}.{code}..Z.processed.annotated.mseed"
            annotated_stream_path = os.path.join(base_dir, 'data', f'{network}.{code}', date_str,
                                                 annotated_stream_file_name)

            annotated_stream = read(annotated_stream_path)
            sliced_annotated_stream = annotated_stream.slice(starttime=starttime, endtime=endtime)

            if len(sliced_annotated_stream) == 0:
                print("No data in the annotated stream.")
                return None

            if ax is None:
                fig, ax = plt.subplots(figsize=(13, 4))

            for pred_trace in sliced_annotated_stream:
                model_name, pred_class = pred_trace.stats.channel.split("_")
                if pred_class == "N":
                    continue  # Skip noise traces
                if p_only and pred_class == "S":
                    continue  # Skip S-wave traces if p_only is True
                c = {"P": "C0", "S": "C1", "De": "#008000"}.get(pred_class, "black")
                offset = pred_trace.stats.starttime - starttime
                label = "Detection" if pred_class == "De" else pred_class
                ax.plot(offset + pred_trace.times(), pred_trace.data, label=label, c=c)
            ax.set_ylabel("Prediction Confidence")
            ax.legend(loc='upper right')
            ax.set_ylim(0, 1.1)

            num_ticks = 5
            x_ticks = np.linspace(0, endtime - starttime, num_ticks)
            x_labels = [(starttime + t).strftime('%H:%M:%S') for t in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=0)

            return ax

        except Exception as e:
            print(f"Error plotting prediction confidence: {e}")
            return None

    def plot_spectrogram(self, starttime, endtime, ax=None):
        """
        Plot the spectrogram for the given time range.

        This method generates a spectrogram plot for the seismic data within the specified
        time range, showing the distribution of frequencies over time.

        Parameters:
        -----------
        starttime : UTCDateTime
            The start time for slicing the stream.
        endtime : UTCDateTime
            The end time for slicing the stream.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axes to plot on. If None, a new figure is created.

        Returns:
        --------
        matplotlib.axes._subplots.AxesSubplot:
            The axis with the plotted spectrogram.
        """
        df_row = self.df_row
        try:
            date_str = df_row['date']
            network = df_row['network']
            code = df_row['code']

            base_dir = os.getcwd()
            stream_file_name = f"{date_str}_{network}.{code}..Z.processed.mseed"
            stream_path = os.path.join(base_dir, 'data', f'{network}.{code}', date_str, stream_file_name)

            stream = read(stream_path)
            trace = stream.slice(starttime=starttime, endtime=endtime)

            if len(trace) == 0:
                print("No data in the trace.")
                return None

            if ax is None:
                fig, ax = plt.subplots(figsize=(13, 4))

            ax.specgram(trace[0].data, NFFT=1024, Fs=trace[0].stats.sampling_rate, noverlap=512, cmap='viridis')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [s]')

            start_time = trace[0].stats.starttime
            end_time = trace[0].stats.endtime
            num_ticks = 5
            x_ticks = np.linspace(0, end_time - start_time, num_ticks)
            x_labels = [(start_time + t).strftime('%H:%M:%S') for t in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=0)

            return ax

        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
            return None

    def plot_event(self, waveform=True, confidence=True, spectrogram=True, p_only=False, deployed=True):
        """
        Plot the event waveform, prediction confidence, and spectrogram, then save the combined plot.

        This method generates a comprehensive plot that includes the waveform, prediction
        confidence, and spectrogram for a seismic event. The plots can be customized to
        include only specific components, and the resulting image is saved locally or uploaded to GitHub.

        Parameters:
        -----------
        waveform : bool, optional
            Whether to plot the waveform.
        confidence : bool, optional
            Whether to plot the prediction confidence.
        spectrogram : bool, optional
            Whether to plot the spectrogram.
        p_only : bool, optional
            If True, only consider P-wave for time range; otherwise, consider both P and S waves.
        deployed : bool, optional
            If True, the plot is uploaded to GitHub; otherwise, it is saved locally.

        Returns:
        --------
        str:
            The path to the saved combined plot, or the GitHub path if deployed.
        """
        df_row = self.df_row
        try:
            if p_only:
                # p_only=True: Only consider P-wave times with fixed 10-second buffers
                p_times = [UTCDateTime(df_row['p_predicted']) if pd.notna(df_row['p_predicted']) else None,
                           UTCDateTime(df_row['p_detected']) if pd.notna(df_row['p_detected']) else None]
                valid_times = [t for t in p_times if t is not None]

                if not valid_times:
                    print("Missing P-wave time data.")
                    return None

                starttime = min(valid_times) - 5
                endtime = max(valid_times) + 5

            else:
                # p_only=False: Consider both P-wave and S-wave times with larger buffers
                p_times = [UTCDateTime(df_row['p_predicted']) if pd.notna(df_row['p_predicted']) else None,
                           UTCDateTime(df_row['p_detected']) if pd.notna(df_row['p_detected']) else None]
                s_times = [UTCDateTime(df_row['s_predicted']) if pd.notna(df_row['s_predicted']) else None,
                           UTCDateTime(df_row['s_detected']) if pd.notna(df_row['s_detected']) else None]
                valid_times = [t for t in p_times + s_times if t is not None]

                if not valid_times:
                    print("Missing P and S wave time data.")
                    return None

                starttime = min(valid_times) - 10
                endtime = max(valid_times) + 10

            starttime = UTCDateTime(starttime)
            endtime = UTCDateTime(endtime)

            num_plots = sum([waveform, confidence, spectrogram])
            fig, axes = plt.subplots(num_plots, 1, figsize=(13, 4 * num_plots), sharex=True)
            if num_plots == 1:
                axes = [axes]

            plot_idx = 0
            if waveform:
                self.plot_waveform(starttime, endtime, ax=axes[plot_idx], p_only=p_only)
                plot_idx += 1

            if confidence:
                self.plot_prediction_confidence(starttime, endtime, ax=axes[plot_idx], p_only=p_only)
                plot_idx += 1

            if spectrogram:
                self.plot_spectrogram(starttime, endtime, ax=axes[plot_idx])

            event_time = pd.to_datetime(df_row['time']).strftime('%Y-%m-%d %H:%M:%S')

            # Reduce the distance between the title and the first subplot
            plt.suptitle(
                f"Earthquake {df_row['unique_id']} - Time: {event_time} - Location: {df_row['lat']:.2f}, {df_row['long']:.2f} - Magnitude: {df_row['mag']} {df_row['mag_type']}",
                fontsize=16, y=0.95)

            plt.subplots_adjust(top=0.90)  # Adjust the top spacing to reduce the gap

            base_dir = os.getcwd()
            network = df_row['network']
            code = df_row['code']
            date_str = df_row['date']
            plot_path = os.path.join(base_dir, 'data', f'{network}.{code}', date_str, 'report')
            os.makedirs(plot_path, exist_ok=True)
            file_name = f'{df_row["unique_id"]}_event_plot.png'
            file_path = os.path.join(plot_path, file_name)

            # Save the plot locally
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)

            if deployed:
                # Define the GitHub path and upload the file
                repo_file_path = os.path.join('data', f'{network}.{code}', date_str, 'report', file_name)
                repo_file_path = repo_file_path.replace("\\", "/")  # Ensure GitHub-compatible path
                upload_file_to_github(file_path, repo_file_path)
                print(f"Uploaded to GitHub: {repo_file_path}")  # Output the uploaded path for debugging
                return repo_file_path

            return file_path

        except Exception as e:
            print(f"Error plotting event: {e}")
            return None
