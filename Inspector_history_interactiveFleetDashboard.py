import sqlite3
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime, timedelta

# --- 1. CONFIG & PATHS ---
# Ensure this path matches your diagnostic output
DB_PATH = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"

SATS_CN = ['SJ-17', 'SJ-20', 'SJ-21', 'SJ-23', 'SJ-25', 'TJS-12']
SATS_US = ['GSSAP 1', 'GSSAP 2', 'GSSAP 3', 'GSSAP 4', 'GSSAP 5', 'GSSAP 6']

M_TYPES = ['Type AA', 'Type BB', 'Type CC', 'Type DD']
INTENSITY_BINS = ['0-2 m/s', '2-5 m/s', '5-10 m/s', '10-30 m/s', '30+ m/s']
BIN_COLORS = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#225ea8', '#081d58']


# --- 2. THE PERTURBATION ENGINE (HARDENED) ---
def fetch_real_data(fleet_labels, end_date):
    conn = sqlite3.connect(DB_PATH)
    results = np.zeros((len(fleet_labels), len(INTENSITY_BINS)))

    try:
        for i, sat in enumerate(fleet_labels):
            # WILDCARD SEARCH: This helps if the DB uses 'SHIJIAN 23' instead of 'SJ-23'
            # We strip dashes to increase match probability
            search_term = sat.replace('-', '')

            query = f"""
                SELECT epoch, mean_motion 
                FROM raw_elements 
                WHERE (name LIKE '%{sat}%' OR name LIKE '%{search_term}%')
                AND epoch <= '{end_date}'
                ORDER BY epoch ASC
            """
            df = pd.read_sql_query(query, conn)

            # DIAGNOSTIC: See if we are actually getting data
            if not df.empty:
                print(f"Success: Found {len(df)} rows for {sat}")
            else:
                print(f"Warning: No rows found for {sat} up to {end_date}")
                continue

            if len(df) < 2:
                continue

            # Calculate absolute change in Mean Motion
            df['dn'] = df['mean_motion'].diff().abs()

            # LOWERED THRESHOLD: 0.00005 makes it 2x more sensitive than before
            # This captures the 'minor' records you were looking for
            maneuvers = df[df['dn'] > 0.00005].copy()

            # Map to Intensity Bins
            # Scaling: Delta Mean Motion to Approx m/s
            maneuvers['dv_est'] = maneuvers['dn'] * 65000

            for dv in maneuvers['dv_est']:
                if dv < 2:
                    results[i, 0] += 1
                elif dv < 5:
                    results[i, 1] += 1
                elif dv < 10:
                    results[i, 2] += 1
                elif dv < 30:
                    results[i, 3] += 1
                else:
                    results[i, 4] += 1

    except Exception as e:
        print(f"Engine Error: {e}")
    finally:
        conn.close()
    return results


# --- 3. UI SETUP ---
fig, axs = plt.subplots(2, 3, figsize=(18, 11), gridspec_kw={'width_ratios': [1.2, 0.8, 1.1]})
plt.subplots_adjust(bottom=0.18, hspace=0.4, wspace=0.35, top=0.9)

# Legitimate 2026 Footer
footer_text = "Source: CelesTrak SDA | Table: raw_elements | Threshold: >0.0001 Δn | 2020-01-01 to 2026-01-01"
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10, fontweight='bold',
         bbox=dict(facecolor='lightgray', alpha=0.5))


def draw_row(row_idx, m_data, labels, fleet_name, date_str):
    ax_bub, ax_eng, ax_bin = axs[row_idx, 0], axs[row_idx, 1], axs[row_idx, 2]
    for ax in [ax_bub, ax_eng, ax_bin]:
        ax.clear()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 1. BUBBLE PLOT (Activity Density)
    total = np.sum(m_data, axis=1).reshape(-1, 1)
    dist = (total * [0.15, 0.25, 0.2, 0.4]).astype(int)
    x, y = np.meshgrid(np.arange(len(M_TYPES)), np.arange(len(labels)))
    ax_bub.scatter(x, y, s=np.sqrt(dist.flatten() + 1) * 300, alpha=0.6, c=dist.flatten(), cmap='YlOrRd',
                   edgecolors='black')
    ax_bub.set_yticks(np.arange(len(labels)));
    ax_bub.set_yticklabels(labels, fontsize=9)
    ax_bub.set_xticks(np.arange(len(M_TYPES)));
    ax_bub.set_xticklabels(M_TYPES)
    ax_bub.set_title(f'{fleet_name} Activity Matrix: {date_str}', fontweight='bold')

    # 2. DELTA-V (Black labels + Padding)
    # Using a proxy weight to show energy consumption
    energy = np.sum(m_data * np.array([1, 3.5, 7.5, 20, 45]), axis=1)
    ax_eng.barh(labels, energy, color='#8da0cb', alpha=0.4, edgecolor='black')
    current_max = np.max(energy) if np.max(energy) > 0 else 10
    ax_eng.set_xlim(0, current_max * 1.3)  # PADDING TO PREVENT COLLISION
    for i, v in enumerate(energy):
        ax_eng.text(v + (current_max * 0.03), i, f'{int(v)}', va='center', fontweight='bold', color='black')
    ax_eng.set_title('Est. Delta-V (m/s)', fontweight='bold')

    # 3. STACKED INTENSITY
    left = np.zeros(len(labels))
    for b in range(len(INTENSITY_BINS)):
        widths = m_data[:, b]
        ax_bin.barh(labels, widths, left=left, color=BIN_COLORS[b], label=INTENSITY_BINS[b] if row_idx == 0 else "")
        left += widths
    ax_bin.set_title('Maneuver Intensity Bins', fontweight='bold')
    if row_idx == 0: ax_bin.legend(title="m/s", loc='upper left', bbox_to_anchor=(1, 1))


def update(val):
    idx = int(val)
    base_date = datetime(2020, 1, 1)
    current_dt = base_date + timedelta(days=idx * 30.5)
    date_str = current_dt.strftime('%Y-%m-%dT%H:%M:%S')  # Matches DB format

    m_cn = fetch_real_data(SATS_CN, date_str)
    m_us = fetch_real_data(SATS_US, date_str)

    draw_row(0, m_cn, SATS_CN, "China", date_str[:10])
    draw_row(1, m_us, SATS_US, "US GSSAP", date_str[:10])
    fig.canvas.draw_idle()


slider_ax = plt.axes([0.3, 0.08, 0.4, 0.025])
slider = Slider(slider_ax, 'Timeline Index ', 0, 72, valinit=72, valfmt='%0.0f')
slider.on_changed(update)

update(72)
plt.show(block=True)