import streamlit as st
import pandas as pd
import requests

# --- 1. PAGE CONFIGURATION ---
# This must be the very first Streamlit command
st.set_page_config(page_title="FL Device Selector", layout="wide")

# --- 2. CUSTOM BABY PINK & WHITE THEME ---
st.markdown("""
    <style>
        /* Main background color */
        .stApp {
            background-color: #FFFFFF;
        }

        /* Sidebar styling with Lavender Blush pink */
        [data-testid="stSidebar"] {
            background-color: #FFF0F5;
        }

        /* Headers and Titles in a deeper rose pink */
        h1, h2, h3 {
            color: #D87093 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Metric values in bright pink */
        [data-testid="stMetricValue"] {
            color: #FF69B4 !important;
        }

        /* Styling buttons to be baby pink with rounded corners */
        .stButton>button {
            background-color: #F4C2C2;
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.5rem 1rem;
        }

        /* Table styling for a cleaner look */
        .stTable {
            background-color: #FFF9FB;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("📱 Federated Learning Device Selection Engine")
st.markdown("Select the best hardware clients for FL training based on real-time API data.")

# --- 3. DATA FETCHING FROM LIVE API ---
@st.cache_data
def load_data():
    # This URL points to your live Vercel Backend
    API_URL = "https://adaptiveclientselectionflapp.vercel.app/get-devices"
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            # Convert API JSON response directly to a DataFrame
            df = pd.DataFrame(response.json())
            # Convert to string to avoid Arrow/JSON serialization errors in the UI
            return df.astype(str)
        else:
            st.error(f"API Error: Received status code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Could not connect to live API: {e}")
        return pd.DataFrame()

# Spinner provides visual feedback while the API processes
with st.spinner('Loading your pink dashboard...'):
    df = load_data()

if not df.empty:
    # --- 4. DATA PRE-PROCESSING ---
    # Extracting numeric values (e.g., '8 GB' -> 8) for filtering logic
    df['RAM_num'] = df['RAM'].str.extract('(\d+)').astype(float)
    df['Bat_num'] = df['Battery Capacity'].str.extract('(\d+)').astype(float)

    # --- 5. SIDEBAR: USER REQUIREMENTS ---
    st.sidebar.header("⚙️ Hardware Requirements")

    req_ram = st.sidebar.number_input("Minimum RAM (GB)", min_value=1, max_value=64, value=8)
    req_battery = st.sidebar.number_input("Minimum Battery (mAh)", min_value=1000, max_value=10000, value=4500)

    # Populate dropdown directly from your data
    processor_options = ["All"] + sorted(df['Processor'].unique().tolist())
    selected_cpu = st.sidebar.selectbox("Select Processor Type", options=processor_options)

    # --- 6. FILTERING LOGIC ---
    mask = (df['RAM_num'] >= req_ram) & (df['Bat_num'] >= req_battery)

    if selected_cpu != "All":
        mask = mask & (df['Processor'] == selected_cpu)

    filtered_df = df[mask].copy()

    # --- 7. OUTPUT DISPLAY ---
    if not filtered_df.empty:
        # Calculate Simulated Accuracy Score for FL suitability
        filtered_df['Accuracy_Score'] = (
                (filtered_df['RAM_num'] * 1.8) +
                (filtered_df['Bat_num'] / 130) +
                35
        ).clip(70.0, 99.2)

        st.subheader(f"✅ Found {len(filtered_df)} Matching Devices")

        # Select a specific device to view details
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
        
        if st.button("Download Selection Report"):
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Click to Download CSV", data=csv, file_name="fl_selection.csv", mime="text/csv")
            
    else:
        st.warning("No devices match your current requirements. Try lowering your minimums!")
else:
    st.info("Please check your Internet connection and Vercel API status.")
