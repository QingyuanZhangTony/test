import os
import warnings

import streamlit as st

from logic import daily_report_automation_logic

from streamlit_utils import load_config_to_session, sidebar_navigation

# 环境变量设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 忽略 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# 加载配置到 session
load_config_to_session()

# Sidebar navigation
sidebar_navigation()

# Streamlit 页面
st.title("Automated Earthquake Data Processing")

# 根据 enable_automation 参数决定是否自动运行处理逻辑
if st.session_state.get('enable_automation', True):
    st.write("Daily Report Automation Enabled.")
    daily_report_automation_logic()
else:
    st.write("Daily Report Automation Disabled.")