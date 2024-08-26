import time
import streamlit as st
from station import Station
from streamlit_utils import load_config_to_df, save_config_to_yaml, load_config_to_session, sidebar_navigation

# Sidebar navigation
sidebar_navigation()

# Load default settings
load_config_to_session()

# Check if it's the first time running
if not st.session_state.get('initialized', False):
    st.title("Welcome to Earthquake Monitoring And Report")
    st.header("First-Time Setup Required")

    st.write("It looks like this is your first time running the application. Please provide the following information to get started.")

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
        # Fetch coordinates based on user inputs
        latitude, longitude = Station.fetch_coordinates(network, station_code, data_provider_url)

        # If coordinates were successfully fetched, proceed
        if latitude is not None and longitude is not None:
            # Update session state with new settings
            st.session_state['network'] = network
            st.session_state['station_code'] = station_code
            st.session_state['data_provider_url'] = data_provider_url
            st.session_state['station_latitude'] = str(latitude)  # Convert to string
            st.session_state['station_longitude'] = str(longitude)  # Convert to string
            st.session_state['email_recipient'] = email_recipient
            st.session_state['initialized'] = True

            # Load and update config file to reflect changes
            config_file = load_config_to_df()
            config_file['network'] = network
            config_file['station_code'] = station_code
            config_file['data_provider_url'] = data_provider_url
            config_file['station_latitude'] = str(latitude)  # Convert to string
            config_file['station_longitude'] = str(longitude)  # Convert to string
            config_file['email_recipient'] = email_recipient
            config_file['initialized'] = True

            # Save the updated config to file
            save_config_to_yaml(config_file)

            # Display success message and prepare for reload
            st.success("Settings saved! Now the application will reload for you to continue.")
            time.sleep(3)  # Wait for 3 seconds before reloading
            st.rerun()  # Reload the application

else:
    # If already initialized, show main content
    st.title("Earthquake Monitoring And Report")
    st.header(f"Welcome back, {st.session_state['station_code']}")

    st.write("You're all set up and ready to start monitoring earthquakes.")

    # Display main application content with formatted coordinates to 2 decimal places
    st.write(f"**Network:** {st.session_state['network']}  **Station:** {st.session_state['station_code']}  **Data Provider URL:** {st.session_state['data_provider_url']}")
    st.write(f"**Data Provider URL:** {st.session_state['data_provider_url']}")
    st.write(f"**Location:** Lat {float(st.session_state['station_latitude']):.2f}, Lon {float(st.session_state['station_longitude']):.2f}")
    st.write(f"**Report Recipient Email:** {st.session_state['email_recipient']}")
