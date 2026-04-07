import streamlit as st
import pandas as pd
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="FL Device Selector", layout="wide")

# --- 2. THE PINK THEME (Added back in) ---
st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; }
        [data-testid="stSidebar"] { background-color: #FFF0F5; }
        h1, h2, h3 { color: #D87093 !important; font-family: sans-serif; }
        [data-testid="stMetricValue"] { color: #FF69B4 !important; }
        .stButton>button { background-color: #F4C2C2; color: white; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📱 Federated Learning Device Selection Engine")
st.markdown("Select the best hardware clients for FL training based on your requirements.")

# --- 3. DATA LOADING (Direct from GitHub) ---
@st.cache_data
def load_data():
    file_path = "devices_dataset.csv.xlsx"
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            return df.astype(str)
        else:
            st.error(f"File '{file_path}' not found.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

df = load_data()

# --- 4. THE REST OF YOUR LOGIC ---
if not df.empty:
    # Numeric Extraction
    df['RAM_num'] = df['RAM'].str.extract('(\d+)').astype(float)
    df['Bat_num'] = df['Battery Capacity'].str.extract('(\d+)').astype(float)

    # Sidebar
    st.sidebar.header("⚙️ Hardware Requirements")
    req_ram = st.sidebar.number_input("Min RAM (GB)", min_value=1, value=
