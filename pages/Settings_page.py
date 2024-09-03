import streamlit as st
from streamlit_utils import load_config_to_df, save_config_to_yaml, sidebar_navigation
from logic import initialisation_logic
import hmac

# Load default settings
default_config = load_config_to_df()

# Initialize or update session state variables from YAML
for key, value in default_config.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Save the initial values of station-related settings for comparison later
initial_values = {
    'network': st.session_state['network'],
    'station_code': st.session_state['station_code'],
    'data_provider_url': st.session_state['data_provider_url'],
    'email_recipient': st.session_state['email_recipient']
}

# Sidebar navigation
sidebar_navigation()

# Page title
st.title("Settings")
st.divider()

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()


# Settings Container
settings_container = st.container()
with settings_container:
    st.header("Settings")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Station Settings", "Catalog Settings", "Stream Processing", "Event Detection", "Report Generation", "Email Settings"])

    # Station Settings Tab
    with tab1:
        network = st.text_input("Network:", value=st.session_state['network'])
        station_code = st.text_input("Station Code:", value=st.session_state['station_code'])

        # Let user select RaspberryShake without the need to enter URL
        data_provider_choice = st.radio(
            "Select Data Provider",
            options=["RaspberryShake", "Custom URL"],
            index=0 if st.session_state['data_provider_url'] == "https://data.raspberryshake.org" else 1
        )

        if data_provider_choice == "RaspberryShake":
            data_provider_url = "https://data.raspberryshake.org"
        else:
            data_provider_url = st.text_input("Custom Data Provider URL:", value=st.session_state['data_provider_url'])

        overwrite = st.checkbox("Overwrite Existing Data", value=st.session_state['overwrite'])

        # 保存选择的数据提供者 URL 到 session state
        st.session_state['data_provider_url'] = data_provider_url


    # Catalog Settings Tab
    with tab2:
        catalog_providers = st.text_input("Catalog Providers:", value=st.session_state['catalog_providers'])
        radmin = st.text_input("Minimum Radius:", value=st.session_state['radmin'])
        radmax = st.text_input("Maximum Radius:", value=st.session_state['radmax'])
        minmag = st.text_input("Minimum Magnitude:", value=st.session_state['minmag'])
        maxmag = st.text_input("Maximum Magnitude:", value=st.session_state['maxmag'])

    # Stream Processing Tab
    with tab3:
        col3, col4 = st.columns(2)
        # Process Stream Settings
        with col3:
            detrend_demean = st.checkbox("Detrend (Demean)", value=st.session_state['detrend_demean'])
            detrend_linear = st.checkbox("Detrend (Linear)", value=st.session_state['detrend_linear'])
            remove_outliers = st.checkbox("Remove Outliers", value=st.session_state['remove_outliers'])
            apply_bandpass = st.checkbox("Bandpass Filter", value=st.session_state['apply_bandpass'])
            taper = st.checkbox("Taper", value=st.session_state['taper'])
            denoise = st.checkbox("Denoise", value=st.session_state['denoise'])

    # Event Detection Tab
    with tab4:
        # Detect Phases Settings
        with st.container():
            st.subheader("Phase Picking Settings")

            p_only = st.checkbox("Use Only P-waves", value=st.session_state['p_only'])
            col1, col2 = st.columns(2)
            with col1:
                p_threshold = st.slider("P Confidence Filtering Threshold", 0.0, 1.0, st.session_state['p_threshold'])
            with col2:
                s_threshold = st.slider("S Confidence Filtering Threshold", 0.0, 1.0, st.session_state['s_threshold'], disabled=p_only)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Match Events Settings
        with st.container():
            st.subheader("Event Matching Settings")
            col3, col4 = st.columns(2)
            with col3:
                tolerance_p = st.slider("P-Phase Matching Tolerance (s)", 0.0, 30.0, st.session_state['tolerance_p'])
            with col4:
                tolerance_s = st.slider("S-Phase Matching Tolerance (s)", 0.0, 30.0, st.session_state['tolerance_s'], disabled=p_only)

    # Report Generation Tab
    with tab5:
        st.subheader("Report Generation")

        simplified = st.checkbox("Simplified Report", value=st.session_state['simplified'])
        fill_map = st.checkbox("Fill Map Style For Catalog Plot", value=st.session_state['fill_map'])

    # Email Settings Tab
    with tab6:
        st.subheader("Email Settings")

        # Use st.columns to adjust the width of the input box
        col1, _ = st.columns([3, 1])
        with col1:
            email_recipient = st.text_input("Recipient Email:", value=st.session_state['email_recipient'])

        st.markdown("<br><br>", unsafe_allow_html=True)

# Save Settings Container
save_settings_container = st.container()
with save_settings_container:
    col1, _ = st.columns(2)
    if col1.button('Save Settings'):
        # Update st.session_state with new settings
        st.session_state.network = network
        st.session_state.station_code = station_code
        st.session_state.data_provider_url = data_provider_url
        st.session_state.catalog_providers = catalog_providers
        st.session_state.radmin = radmin
        st.session_state.radmax = radmax
        st.session_state.minmag = minmag
        st.session_state.maxmag = maxmag
        st.session_state.detrend_demean = detrend_demean
        st.session_state.detrend_linear = detrend_linear
        st.session_state.remove_outliers = remove_outliers
        st.session_state.apply_bandpass = apply_bandpass
        st.session_state.taper = taper
        st.session_state.denoise = denoise
        st.session_state.p_only = p_only
        st.session_state.p_threshold = p_threshold
        st.session_state.s_threshold = s_threshold
        st.session_state.tolerance_p = tolerance_p
        st.session_state.tolerance_s = tolerance_s
        st.session_state.email_recipient = email_recipient
        st.session_state.overwrite = overwrite
        st.session_state.simplified = simplified
        st.session_state.fill_map = fill_map

        # 检查站台相关值是否发生变化
        if (
            initial_values['network'] != st.session_state.network or
            initial_values['station_code'] != st.session_state.station_code or
            initial_values['data_provider_url'] != st.session_state.data_provider_url or
            initial_values['email_recipient'] != st.session_state.email_recipient
        ):
            # 调用初始化逻辑函数
            initialisation_logic(
                st.session_state.network,
                st.session_state.station_code,
                st.session_state.data_provider_url,
                st.session_state.email_recipient,
                max_attempts=3
            )

        # Save the updated config to the file
        save_config_to_yaml(st.session_state)

        # Refresh the page to avoid widget state conflict
        st.rerun()
