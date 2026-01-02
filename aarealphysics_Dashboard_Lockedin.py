import sqlite3
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from datetime import datetime, timedelta

# --- 1. GLOBALS & CONFIG ---
DB_PATH = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 1, 1)
TOTAL_DAYS = (END_DATE - START_DATE).days

M_CLASSES = ['Station Keeping', 'Ingress/Phasing', 'Plane Change', 'Aggressive/RPO']
INTENSITY_BINS = ['0-2 m/s', '2-5 m/s', '5-10 m/s', '10-50 m/s', '50-100 m/s']
INTENSITY_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']

FLEET_ORDER = [
    (46610, 'GF-13', 'Oct 2020'), (58957, 'TJS-11', 'Feb 2024'),
    (41838, 'SJ-17', 'Nov 2016'), (49330, 'SJ-21', 'Oct 2021'),
    (55222, 'SJ-23', 'Jan 2023'), (62485, 'SJ-25', 'Jan 2025')
]

SIGNIFICANT_EVENTS = [
    {'date': datetime(2020, 10, 11), 'label': '^ GF-13 Launch', 'detail': 'High-orbit optical debut.'},
    {'date': datetime(2021, 10, 24), 'label': '^ SJ-21 Launch', 'detail': 'Tug tech demonstrator.'},
    {'date': datetime(2022, 1, 22), 'label': '* SJ-21 RPO', 'detail': 'Docking/Tow of Compass-G2.'},
    {'date': datetime(2023, 1, 13), 'label': '^ SJ-23 Launch', 'detail': 'Classified multi-payload.'},
    {'date': datetime(2024, 2, 23), 'label': '^ TJS-11 Launch', 'detail': 'Comm-tech test satellite.'},
    {'date': datetime(2025, 1, 10), 'label': '^ SJ-25 Launch', 'detail': 'Advanced debris mission.'}
]


# --- 2. DATA ENGINE ---
def fetch_operational_data(end_date_str):
    conn = sqlite3.connect(DB_PATH)
    type_matrix = np.zeros((len(FLEET_ORDER), 4))
    intensity_matrix = np.zeros((len(FLEET_ORDER), 5))
    dv_sums = []
    labels = [f"{item[1]}\n{item[2]}" for item in FLEET_ORDER]

    for i, (scn, name, date_str) in enumerate(FLEET_ORDER):
        df = pd.read_sql_query(
            f"SELECT mean_motion FROM gp_history WHERE norad_id={scn} AND epoch <= '{end_date_str}' ORDER BY epoch ASC",
            conn)
        if len(df) < 5:
            dv_sums.append(0);
            continue

        df['dv_ms'] = df['mean_motion'].diff().abs() * 65000
        # UPDATED FILTER: Exclude <0.5 (noise) and >100 (sensor/TLE error)
        burns = df[(df['dv_ms'] >= 0.5) & (df['dv_ms'] <= 100.0)].copy()
        dv_sums.append(burns['dv_ms'].sum())

        for dv in burns['dv_ms']:
            if dv < 2:
                intensity_matrix[i, 0] += 1
            elif dv < 5:
                intensity_matrix[i, 1] += 1
            elif dv < 10:
                intensity_matrix[i, 2] += 1
            elif dv < 50:
                intensity_matrix[i, 3] += 1
            else:
                intensity_matrix[i, 4] += 1

            if dv < 1.5:
                type_matrix[i, 0] += 1
            elif dv < 5.0:
                type_matrix[i, 1] += 1
            elif dv < 20.0:
                type_matrix[i, 2] += 1
            else:
                type_matrix[i, 3] += 1
    conn.close()
    return labels[::-1], type_matrix[::-1], intensity_matrix[::-1], dv_sums[::-1]


# --- 3. UI SETUP ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), facecolor='#0d1117')
# Increased wspace to 0.45 to prevent AX3 from hitting the AX2 legend
plt.subplots_adjust(top=0.82, bottom=0.30, wspace=0.45, left=0.08, right=0.92)

# TOP: Dashboard Title & Audit Range
fig.suptitle("CHINESE GEO ASSET MANEUVER ANALYTICS", color='white', fontsize=22, fontweight='bold', y=0.97)
scope_note = fig.text(0.5, 0.90, "", ha='center', color='#8b949e', fontsize=12, fontweight='bold')

# BOTTOM: Instructions & Metadata (Updated Period and Filter)
event_popup = fig.text(0.5, 0.06, "Click a Diamond on the Timeline below to jump to that event", ha='center',
                       color='#58a6ff', fontsize=11, fontweight='bold')
info_text = fig.text(0.5, 0.02,
                     "SOURCE: celestrak.org | FILTER: Maneuvers < 0.5 m/s and > 100 m/s ignored | PERIOD: Jan 2020 - Dec 2025",
                     ha='center', color='gray', fontsize=8, family='monospace')

# Timeline Axis
ax_time = plt.axes([0.15, 0.10, 0.7, 0.015], facecolor='#161b22')
ax_time.set_xlim(START_DATE, END_DATE)
ax_time.tick_params(axis='x', colors='#8b949e', labelsize=9)
ax_time.get_yaxis().set_visible(False)

for ev in SIGNIFICANT_EVENTS:
    day_offset = (ev['date'] - START_DATE).days
    point, = ax_time.plot(ev['date'], 0, 'D', color='#f85149', markersize=10, picker=True, pickradius=12, clip_on=False)
    point.set_gid(day_offset)
    point.set_label(f"{ev['label']}: {ev['detail']}")


# --- 4. UPDATE LOGIC ---
def update(val):
    current_dt = START_DATE + timedelta(days=int(val))
    labels, type_data, int_data, dv_sums = fetch_operational_data(current_dt.strftime('%Y-%m-%d'))

    # AX1: Maneuver Classes
    ax1.clear();
    ax1.set_facecolor('#161b22')
    for i in range(len(labels)):
        for j in range(4):
            count = type_data[i, j]
            if count > 0:
                ax1.scatter(j, i, s=np.sqrt(count) * 320, color='#58a6ff', alpha=0.5, edgecolors='white')
                ax1.text(j, i, int(count), color='white', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(4));
    ax1.set_xticklabels(M_CLASSES, color='#8b949e', rotation=25, ha='right')
    ax1.set_yticks(range(len(labels)));
    ax1.set_yticklabels(labels, color='white', fontweight='bold')
    ax1.set_title("MANEUVER CLASS COUNT", color='white', pad=20)

    # AX2: Intensity (No Y-Labels)
    ax2.clear();
    ax2.set_facecolor('#161b22')
    ax2.set_yticklabels([])
    bottom = np.zeros(len(labels))
    for b in range(5):
        ax2.barh(labels, int_data[:, b], left=bottom, color=INTENSITY_COLORS[b], label=INTENSITY_BINS[b])
        bottom += int_data[:, b]
    for i, total in enumerate(bottom):
        ax2.text(total + 0.5, i, f"Sum: {int(total)}", color='white', va='center', fontsize=9)
    ax2.set_title("INTENSITY BY BIN", color='white', pad=20)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, labelcolor='white',
               title="Magnitude (m/s)")

    # AX3: Delta-V (No Y-Labels)
    ax3.clear();
    ax3.set_facecolor('#161b22')
    ax3.set_yticklabels([])
    ax3.barh(labels, dv_sums, color='#f85149', alpha=0.8)
    max_dv = max(dv_sums) if len(dv_sums) > 0 and max(dv_sums) > 0 else 3000
    ax3.set_xlim(0, max_dv * 1.35)
    ax3.set_title("TOTAL MISSION Delta-V", color='white', pad=20)
    ax3.set_xlabel("Cumulative m/s", color='#8b949e', fontsize=10)
    ax3.tick_params(axis='x', colors='#8b949e', labelsize=9)
    for i, v in enumerate(dv_sums):
        ax3.text(v + (max_dv * 0.03), i, f"{int(v)}", color='white', va='center', fontsize=10, fontweight='bold')

    scope_note.set_text(f"AUDIT WINDOW: 01 JAN 2020 - {current_dt.strftime('%d %b %Y').upper()}")
    fig.canvas.draw_idle()


def on_pick(event):
    slider.set_val(event.artist.get_gid())
    event_popup.set_text(f"SYNCED TO: {event.artist.get_label()}")
    event_popup.set_color('#ff7b72')


fig.canvas.mpl_connect('pick_event', on_pick)

# --- 5. CONTROLS ---
# SLIDER BAR: Moved down from 0.22 to 0.18 to clear the Bubble Plot X-labels
slider_ax = plt.axes([0.3, 0.18, 0.4, 0.025], facecolor='#30363d')
slider = Slider(slider_ax, 'Timeline', 0, TOTAL_DAYS, valinit=TOTAL_DAYS, valstep=1)
slider.on_changed(update)

btn_prev = Button(plt.axes([0.24, 0.18, 0.04, 0.025], facecolor='#21262d'), '<', color='#21262d')
btn_next = Button(plt.axes([0.72, 0.18, 0.04, 0.025], facecolor='#21262d'), '>', color='#21262d')

btn_prev.on_clicked(lambda e: slider.set_val(max(0, slider.val - 30)))
btn_next.on_clicked(lambda e: slider.set_val(min(TOTAL_DAYS, slider.val + 30)))

update(TOTAL_DAYS)
plt.show()