import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FL Device Selector", layout="wide")

st.title("📱 Federated Learning Device Selection Engine")
st.markdown("""
This dashboard selects the best hardware clients for Federated Learning training 
based on your specific hardware requirements.
""")


# --- DATA FETCHING ---
@st.cache_data
def load_data():
    try:
        # Connecting to your FastAPI backend running in Docker
        response = requests.get("http://localhost:8000/get-devices")
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            # Ensure all data is treated as strings to prevent ArrowTypeError
            return data.astype(str)
        else:
            st.error("Backend reached but returned an error.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Could not connect to Backend: {e}")
        return pd.DataFrame()


df = load_data()

if not df.empty:
    # --- DATA PRE-PROCESSING ---
    # Convert strings like '8 GB' and '5000 mAh' into numbers for filtering
    df['RAM_num'] = df['RAM'].str.extract('(\d+)').astype(float)
    df['Bat_num'] = df['Battery Capacity'].str.extract('(\d+)').astype(float)

    # --- SIDEBAR: USER REQUIREMENTS ---
    st.sidebar.header("⌨️ Set Hardware Requirements")

    # Manual Number Inputs
    req_ram = st.sidebar.number_input("Minimum RAM (GB)", min_value=1, max_value=64, value=8)
    req_battery = st.sidebar.number_input("Minimum Battery (mAh)", min_value=1000, max_value=10000, value=4500)

    # Dynamic Processor Dropdown
    # Gets unique processors from your Excel file automatically
    processor_options = ["All"] + sorted(df['Processor'].unique().tolist())
    selected_cpu = st.sidebar.selectbox("Select Processor Type", options=processor_options)

    # --- FILTERING LOGIC ---
    mask = (df['RAM_num'] >= req_ram) & (df['Bat_num'] >= req_battery)

    if selected_cpu != "All":
        mask = mask & (df['Processor'] == selected_cpu)

    filtered_df = df[mask].copy()

    # --- OUTPUT DISPLAY ---
    if not filtered_df.empty:
        # Calculate Simulated Accuracy Score (Hardware stability for FL)
        # Better specs = Higher potential accuracy
        filtered_df['Accuracy_Score'] = (
                (filtered_df['RAM_num'] * 1.8) +
                (filtered_df['Bat_num'] / 130) +
                35
        ).clip(70.0, 99.2)

        st.subheader(f"✅ Found {len(filtered_df)} Matching Devices")

        # Select specific device to see features
        selected_model = st.selectbox("Select a device for full feature output:", filtered_df['Model Name'])
        device_data = filtered_df[filtered_df['Model Name'] == selected_model].iloc[0]

        # Final Feature Output Cards
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.header(f"✨ {device_data['Model Name']}")
            st.metric(label="Predicted FL Accuracy", value=f"{float(device_data['Accuracy_Score']):.2f}%")
            st.write(f"**Manufacturer:** {device_data['Company Name']}")
            st.write(f"**Processor:** {device_data['Processor']}")

        with col2:
            st.info("📋 Technical Specifications")
            # Cleaning up the display table
            specs = {
                "RAM": device_data['RAM'],
                "Battery": device_data['Battery Capacity'],
                "Weight": device_data.get('Mobile Weight', 'N/A'),
                "Storage": device_data.get('Internal Storage', 'N/A')
            }
            st.table(pd.DataFrame(specs.items(), columns=["Feature", "Value"]))

        st.success("Analysis complete. This device is eligible for the Federated Learning cluster.")

    else:
        st.warning("No devices match your current requirements. Try lowering the RAM or Battery values.")

else:
    st.info("Waiting for data from API... Ensure your Docker containers are running.")