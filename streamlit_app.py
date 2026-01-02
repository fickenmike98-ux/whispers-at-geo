import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Chinese GEO Analytics", layout="wide", initial_sidebar_state="expanded")

# --- 2. GLOBALS & METADATA ---
DB_PATH = "celestrak_public_sda.db"
FLEET_METADATA = {
    41838: {"name": "SJ-17", "launch": "2016-11-03"},
    44387: {"name": "SJ-20", "launch": "2019-12-27"},
    49330: {"name": "SJ-21", "launch": "2021-10-24"},
    55222: {"name": "SJ-23", "launch": "2023-01-08"},
    62485: {"name": "SJ-25", "launch": "2024-10-23"},
    58957: {"name": "TJS-11", "launch": "2024-02-23"},
    46610: {"name": "GF-13", "launch": "2020-10-11"}
}
FLEET_ORDER = [46610, 58957, 41838, 49330, 55222, 62485]


@st.cache_data
def fetch_data(end_date_str, mode):
    if not os.path.exists(DB_PATH): return [], 0
    conn = sqlite3.connect(DB_PATH)
    results = []
    global_total_records = 0
    low_floor = 0.15 if mode == "STANDARD" else 0.04
    MU = 398600.4418

    for scn in FLEET_ORDER:
        meta = FLEET_METADATA.get(scn, {"name": f"Unknown ({scn})", "launch": "N/A"})
        query = f"SELECT mean_motion, epoch FROM gp_history WHERE norad_id={scn} AND epoch <= '{end_date_str}' ORDER BY epoch ASC"
        df = pd.read_sql_query(query, conn)

        type_bins, int_bins = [0] * 4, [0] * 5
        total_dv, n_obs = 0, len(df)
        global_total_records += n_obs

        if n_obs >= 5:
            df['n_smooth'] = df['mean_motion'].rolling(window=3, center=True).median().ffill().bfill()
            n_rad_s = (df['n_smooth'] * 2 * np.pi) / 86400
            df['sma'] = (MU / (n_rad_s ** 2)) ** (1 / 3)
            df['dv_total'] = (3.0747 / (2 * df['sma'])) * df['sma'].diff().abs() * 1000

            burns = df[(df['dv_total'] >= low_floor) & (df['dv_total'] <= 100.0)].copy()
            total_dv = burns['dv_total'].sum()

            for dv in burns['dv_total']:
                if dv < 1:
                    int_bins[0] += 1
                elif dv < 2:
                    int_bins[1] += 1
                elif dv < 5:
                    int_bins[2] += 1
                elif dv < 15:
                    int_bins[3] += 1
                else:
                    int_bins[4] += 1

                if dv < 1.2:
                    type_bins[0] += 1
                elif dv < 4.0:
                    type_bins[1] += 1
                elif dv < 15.0:
                    type_bins[2] += 1
                else:
                    type_bins[3] += 1

        results.append({
            'name': meta['name'], 'launch': meta['launch'], 'n_obs': n_obs,
            'type_counts': type_bins, 'int_counts': int_bins, 'total_dv': total_dv,
            'total_maneuvers': sum(int_bins)
        })
    conn.close()
    return results, global_total_records


# --- 3. SIDEBAR ---
st.sidebar.title("🛠️ Dashboard Controls")
analysis_date = st.sidebar.slider("Timeline Explorer", datetime(2020, 1, 1), datetime(2026, 1, 1), datetime(2026, 1, 1),
                                  format="DD MMM YYYY")
mode = st.sidebar.radio("Sensitivity Mode", ["STANDARD", "SENSITIVE"])

data, total_n = fetch_data(analysis_date.strftime('%Y-%m-%d'), mode)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 DATA SCIENCE NOTES")

st.sidebar.warning(f"""
**DATA SOURCE:**
- **Provider:** Celestrak.org
- **Epoch:** 2020 - 2025
- **Total Records:** {total_n:,} TLEs
""")

st.sidebar.markdown(f"""
**DATA CLEANING:**
Public TLE data often contains 'jitter' or small errors. This used median filtering to smooth noise and reduce (prevent) false maneuver detections.

**GVE DEFINED:**
Gauss Variational Equations (GVE) is the approach used to turn orbital changes into propulsion estimates. They calculate how a 'push' in one direction alters the satellite's overall path.

**PROPAGATOR SELECTION:**
SGP4 is used because public TLEs are specifically built for it. Using SGP8 would create math errors due to analytical inconsistency.
""")

# --- NEW POC SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("📬 Point of Contact")
st.sidebar.markdown("**Michael Ficken**")
st.sidebar.link_button("Connect on LinkedIn", "https://www.linkedin.com/in/fickenmike/")

# --- 4. MAIN DASHBOARD ---
st.title("CHINESE GEO ASSET MANEUVERS - A LOOK INTO THE PAST")

if data:
    y_labels = [f"<b>{d['name']}</b><br>L: {d['launch']}<br>N={d['n_obs']}" for d in data][::-1]

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.12,
                        subplot_titles=("TACTICAL EVENT TYPE", "INTENSITY (STACKED BINS)", "TOTAL MISSION ΔV (m/s)"))

    # Plot 1: Bubble Chart
    types = ['Station Keep', 'Ingress', 'Plane Change', 'Aggressive']
    for i, d in enumerate(data[::-1]):
        sizes = [np.sqrt(x) * 22 if x > 0 else 0 for x in d['type_counts']]
        fig.add_trace(go.Scatter(
            x=types, y=[y_labels[i]] * 4, mode='markers+text',
            marker=dict(size=sizes, color='#58a6ff', opacity=0.7),
            text=[int(x) if x > 0 else "" for x in d['type_counts']],
            textposition="middle center", showlegend=False), row=1, col=1)

    # Plot 2: Intensity (Stacked)
    int_labels = ['0-1m/s', '1-2m/s', '2-5m/s', '5-15m/s', '15+m/s']
    colors = ['#2ecc71', '#82e0aa', '#f1c40f', '#e67e22', '#e74c3c']
    for b_idx in range(5):
        fig.add_trace(go.Bar(
            y=y_labels, x=[d['int_counts'][b_idx] for d in data[::-1]],
            name=int_labels[b_idx], orientation='h', marker_color=colors[b_idx],
            legendgroup="int",
            text=[f"<b>{d['total_maneuvers']}</b>" if b_idx == 4 else "" for d in data[::-1]],
            textposition='outside', cliponaxis=False), row=1, col=2)

    # Plot 3: Mission Total ΔV
    fig.add_trace(go.Bar(
        y=y_labels, x=[round(d['total_dv'], 2) for d in data[::-1]],
        orientation='h', marker_color='#f85149', showlegend=False,
        text=[f"{round(d['total_dv'], 1)}" for d in data[::-1]], textposition='auto'), row=1, col=3)

    fig.update_layout(
        template="plotly_dark",
        height=950,
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=150, r=50, t=100, b=100)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Database connection failed or no data found.")