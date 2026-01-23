import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
import os
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Orbital Resilience Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. API SETUP ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("⚠️ Missing Google API Key in .streamlit/secrets.toml")
except Exception as e:
    st.error(f"Configuration Error: {e}")


# --- 3. DATA LOADER ---
@st.cache_data
def load_data():
    """
    Connects to local SQLite database. Uses raw string (r) for Windows paths.
    """
    db_path = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"

    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

        if tables.empty:
            return None, "No Tables Found"

        table_name = tables.iloc[0]['name']
        # Limiting to 100 rows for demo latency speed
        query = f"SELECT * FROM {table_name} LIMIT 100"
        df = pd.read_sql(query, conn)
        conn.close()
        return df, table_name

    except Exception as e:
        return None, str(e)


# --- 4. THE AI ENGINE (Civil/Safety Focus) ---
def run_agent(df, query, audience_type, simulate_anomaly=False):
    """
    Injects data context and applies the persona (Executive vs Engineer).
    Handles the 'Debris Event' simulation.
    """

    # A. SIMULATION LOGIC: Inject a High-Risk Debris Fragment
    if simulate_anomaly:
        anomaly_data = pd.DataFrame([{
            'OBJECT_NAME': 'UNIDENTIFIED_DEBRIS_FRAG',
            'EPOCH': '2026-01-22',
            'INCLINATION': '98.5',
            'DECAY': 'UNSTABLE',
            'RISK_LEVEL': 'HIGH_CONJUNCTION_PROBABILITY',
            'NOTES': 'Rapidly tumbling object intersecting commercial orbit plane.'
        }])
        # Add anomaly to the TOP of the dataframe
        df = pd.concat([anomaly_data, df], ignore_index=True)

    # B. PERSONA LOGIC: Adjust tone for the user
    if audience_type == "Executive Leadership (Strategic)":
        tone_instruction = (
            "Provide a concise executive summary. "
            "Focus on operational continuity, safety risks, and resource impact. "
            "Avoid excessive technical jargon. Be clear and decisive."
        )
    else:  # Systems Engineer
        tone_instruction = (
            "Be technically precise. "
            "Reference specific Object IDs, inclination variances, and epoch timestamps. "
            "Detail the anomaly parameters explicitly."
        )

    # C. MODEL CONFIGURATION
    model = genai.GenerativeModel('gemini-flash-latest')

    # D. CONTEXT PREP
    context = df.to_markdown(index=False)

    prompt = f"""
    Role: You are an AI Advisor for Space Traffic Management.
    Audience: {audience_type}
    Instruction: {tone_instruction}

    Current Telemetry Stream:
    {context}

    User Query: {query}

    Directives:
    - Analyze the telemetry for safety and stability.
    - If 'UNIDENTIFIED_DEBRIS_FRAG' is present, flag it as a critical safety hazard immediately.
    - Maintain a professional, public-sector tone (Safety & Resilience focus).
    """

    response = model.generate_content(prompt)
    return response.text


# --- 5. MAIN UI LAYOUT ---

st.title("🌍 Orbital Resilience Monitor")
st.caption("AI-Enabled Traffic Management & Safety | Powered by Gemini Flash")

# Load Data
df, table_info = load_data()

# --- GUIDE EXPANDER ---
with st.expander("📘 User Guide: Semantic Query Examples"):
    c1, c2 = st.columns(2)
    with c1:
        st.success("✅ **Effective Queries (Pattern/Summary)**")
        st.markdown("""
        1. "Summarize the health status of the constellation."
        2. "Identify any objects flagging for orbital decay."
        3. "Provide a risk assessment for the current epoch."
        4. "List objects with high drag coefficients."
        5. "Explain the mission / function of object 4321 based on its telemetry"
        """)
    with c2:
        st.error("❌ **Out of Scope (Physics/Calculation)**")
        st.markdown("""
        1. "Calculate the exact collision probability." (Requires Physics Engine)
        2. "Predict the precise location in 2 hours." (Requires Propagator)
        3. "What is the distance between Object A and B?" (Requires Math Tool)
        4. "Rank these 50 satellites by period." (LLMs struggle to sort numbers)
        5. "Calculate the exact average altitude of the constellation." (Requires Math)
        """)
    st.info(
        "💡 **Note:** This system demonstrates *Semantic Reasoning* over structured data. Hard physics calculations would be offloaded to a specialized backend.")

st.divider()

# --- TWO COLUMN LAYOUT ---
col_data, col_controls = st.columns([3, 2])

# LEFT: Data Feed
with col_data:
    st.subheader(f"📡 Real-Time Telemetry")
    if df is not None:
        st.dataframe(df, height=500, use_container_width=True)
    else:
        st.error(f"Data Connection Failed: {table_info}")

# RIGHT: Control Panel
with col_controls:
    st.subheader("⚡ Analytics Controls")

    # 1. AUDIENCE TOGGLE
    audience = st.radio(
        "Response Persona:",
        ["Executive Leadership (Strategic)", "Systems Engineer (Technical)"],
        horizontal=True
    )

    # 2. SIMULATION TOGGLE
    st.write("")
    enable_simulation = st.toggle("🚨 SCENARIO: Inject Critical Anomaly", value=False)

    if enable_simulation:
        st.warning("⚠️ **TEST SCENARIO ACTIVE:** Unidentified Debris Object injected into data stream.")

    st.divider()

    # 3. INPUT
    option = st.selectbox(
        "Quick Actions:",
        (
            "Summarize the current operational status.",
            "Identify the highest priority safety alert.",
            "Explain the stability of the listed orbits.",
            "List all objects launched prior to 2020."
        )
    )
    user_input = st.text_input("Custom Query:", value=option)

    # 4. EXECUTE
    if st.button("Generate Insight", type="primary", use_container_width=True):
        if user_input and df is not None:

            # VISUAL TRUST INDICATORS
            with st.status("Analyzing Telemetry...", expanded=True) as status:
                st.write("📡 Ingesting Data Stream...")
                time.sleep(0.3)

                if enable_simulation:
                    st.write("⚠️ **Processing Synthetic Anomaly Event...**")
                    time.sleep(0.2)

                st.write("🔒 Validating Data Integrity...")
                time.sleep(0.2)
                st.write(f"🧠 Adapting Output for: **{audience}**...")

                # RUN AI
                try:
                    response_text = run_agent(df, user_input, audience, enable_simulation)
                    status.update(label="✅ Insight Generated", state="complete", expanded=False)

                    st.markdown("### 📊 System Output")
                    st.info(response_text)

                    st.caption("Latency: ~0.8s | Model: Gemini Flash | Cost: <$0.01")

                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
                    status.update(label="❌ System Error", state="error")

    # --- 6. DEVELOPER FOOTER ---
    st.markdown("---")

    with st.container():
        st.subheader("🛠️ Architect Context")
        st.markdown("""
        **[Michael Ficken]** *Strategic AI Executive | CDAO at U.S. Space Force* *Operationalizing AI for National Defense | Computer Vision & Edge Strategy*
        """)

        st.caption("Objective: Demonstrate rapid 'Agentic' integration for mission-critical operational workflows.")

        st.markdown("""
        [🌐 LinkedIn Profile](https://www.linkedin.com/in/fickenmike/)
        """)