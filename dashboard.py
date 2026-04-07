import streamlit as st
import pandas as pd
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="FL Device Selector", layout="wide")

# --- 2. EXACT ORIGINAL THEME (PINK & ROSE) ---
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #FFFFFF;
        }

        /* Sidebar: Lavender Blush */
        [data-testid="stSidebar"] {
            background-color: #FFF0F5;
        }

        /* Headers: Deep Rose Pink */
        h1, h2, h3 {
            color: #D87093 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Metric values: Bright Pink */
        [data-testid="stMetricValue"] {
            color: #FF69B4 !important;
        }

        /* Sidebar labels and text */
        .css-17lntkn { 
            color: #D87093; 
        }

        /* Success message styling */
        .stSuccess {
            background-color: #F0FFF0;
            color: #2E8B57;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("📱 Federated Learning Device Selection Engine")
st.markdown("Select the best hardware clients for FL training based on your specific requirements.")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    file_path = "devices_dataset.csv.xlsx"
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            # Ensure everything is a string to match documentation outputs
            return df.astype(str)
        else:
            st.error(f"File '{file_path}' not found in repository.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # --- 4. DATA PROCESSING ---
    df['RAM_num'] = df['RAM'].str.extract('(\d+)').astype(float)
    df['Bat_num'] = df['Battery Capacity'].str.extract('(\d+)').astype(float)

    # --- 5. SIDEBAR: HARDWARE REQUIREMENTS ---
    st.sidebar.header("⚙️ Hardware Requirements")
    req_ram = st.sidebar.number_input("Minimum RAM (GB)", min_value=1, max_value=64, value=8)
    req_battery = st.sidebar.number_input("Minimum Battery (mAh)", min_value=1000, max_value=10000, value=4500)

    processor_options = ["All"] + sorted(df['Processor'].unique().tolist())
    selected_cpu = st.sidebar.selectbox("Select Processor Type", options=processor_options)

    # --- 6. FILTERING ---
    mask = (df['RAM_num'] >= req_ram) & (df['Bat_num'] >= req_battery)
    if selected_cpu != "All":
        mask = mask & (df['Processor'] == selected_cpu)

    filtered_df = df[mask].copy()

    # --- 7. OUTPUT DISPLAY ---
    if not filtered_df.empty:
        # Exact calculation used in your documentation
        filtered_df['Accuracy_Score'] = (
            (filtered_df['RAM_num'] * 1.8) + (filtered_df['Bat_num'] / 130) + 35
        ).clip(70.0, 99.2)

        st.subheader(f"✅ Found {len(filtered_df)} Matching Devices")

        selected_model = st.selectbox("Select a device for full details:", filtered_df['Model Name'])
        device_data = filtered_df[filtered_df['Model Name'] == selected_model].iloc[0]

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.header(f"✨ {device_data['Model Name']}")
            st.metric(label="Predicted FL Accuracy", value=f"{float(device_data['Accuracy_Score']):.2f}%")
            st.write(f"**Manufacturer:** {device_data['Company Name']}")
            st.write(f"**Processor:** {device_data['Processor']}")

        with col2:
            st.info("📋 Technical Specifications")
            specs = {
                "RAM": device_data['RAM'],
                "Battery": device_data['Battery Capacity'],
                "Weight": device_data.get('Mobile Weight', 'N/A'),
                "Internal Storage": device_data.get('Internal Storage', 'N/A')
            }
            st.table(pd.DataFrame(specs.items(), columns=["Feature", "Value"]))

        st.success("Analysis complete. This device is eligible for the Federated Learning cluster.")
    else:
        st.warning("No devices match your current requirements.")
