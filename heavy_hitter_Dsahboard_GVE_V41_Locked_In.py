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
START_DATE, END_DATE = datetime(2020, 1, 1), datetime(2026, 1, 1)
TOTAL_DAYS = (END_DATE - START_DATE).days

M_CLASSES = ['Station Keeping', 'Ingress/Phasing', 'Plane Change', 'Aggressive/RPO']
INTENSITY_BINS = ['0-1 m/s', '1-2 m/s', '2-5 m/s', '5-15 m/s', '15+ m/s']
INTENSITY_COLORS = ['#2ecc71', '#82e0aa', '#f1c40f', '#e67e22', '#e74c3c']

FLEET_ORDER = [(46610, 'GF-13', 'Oct 2020'), (58957, 'TJS-11', 'Feb 2024'),
               (41838, 'SJ-17', 'Nov 2016'), (49330, 'SJ-21', 'Oct 2021'),
               (55222, 'SJ-23', 'Jan 2023'), (62485, 'SJ-25', 'Jan 2025')]

SIGNIFICANT_EVENTS = [
    {'date': datetime(2020, 10, 11), 'label': '^ GF-13 Launch', 'detail': 'High-orbit optical debut.'},
    {'date': datetime(2021, 10, 24), 'label': '^ SJ-21 Launch', 'detail': 'Tug tech demonstrator.'},
    {'date': datetime(2022, 1, 22), 'label': '* SJ-21 RPO', 'detail': 'Docking/Tow of Compass-G2.'},
    {'date': datetime(2023, 1, 13), 'label': '^ SJ-23 Launch', 'detail': 'Classified multi-payload.'},
    {'date': datetime(2024, 2, 23), 'label': '^ TJS-11 Launch', 'detail': 'Comm-tech test satellite.'},
    {'date': datetime(2025, 1, 10), 'label': '^ SJ-25 Launch', 'detail': 'Advanced debris mission.'}
]

SENSITIVITY_MODE = "STANDARD"
CI_OVERLAY_ACTIVE = False


# --- 2. DATA ENGINE ---
def fetch_operational_data(end_date_str, mode):
    conn = sqlite3.connect(DB_PATH)
    type_matrix = np.zeros((len(FLEET_ORDER), 4))
    intensity_matrix = np.zeros((len(FLEET_ORDER), 5))
    dv_sums, obs_raw, labels = [], [], []

    low_floor = 0.15 if mode == "STANDARD" else 0.04
    MU = 398600.4418

    for i, (scn, name, date_str) in enumerate(FLEET_ORDER):
        df = pd.read_sql_query(
            f"SELECT mean_motion, inclination FROM gp_history WHERE norad_id={scn} AND epoch <= '{end_date_str}' ORDER BY epoch ASC",
            conn)
        n_obs = len(df)
        obs_raw.append(n_obs)
        labels.append(f"{name}\n{date_str}\nObs: {n_obs}")

        if n_obs < 5: dv_sums.append(0); continue

        df['n_smooth'] = df['mean_motion'].rolling(window=3, center=True).median().ffill().bfill()
        df['i_smooth'] = df['inclination'].rolling(window=3, center=True).median().ffill().bfill()
        n_rad_s = (df['n_smooth'] * 2 * np.pi) / 86400
        df['sma'] = (MU / (n_rad_s ** 2)) ** (1 / 3)
        df['dv_in_plane'] = (3.0747 / (2 * df['sma'])) * df['sma'].diff().abs() * 1000
        df['di_rad'] = np.radians(df['i_smooth'].diff().abs())
        df['dv_out_plane'] = 2 * 3.0747 * np.sin(df['di_rad'] / 2) * 1000
        df['dv_total'] = df['dv_in_plane'].fillna(0) + df['dv_out_plane'].fillna(0)

        burns = df[(df['dv_total'] >= low_floor) & (df['dv_total'] <= 100.0)].copy()
        dv_sums.append(burns['dv_total'].sum())
        for dv in burns['dv_total']:
            idx_int = 0 if dv < 1 else 1 if dv < 2 else 2 if dv < 5 else 3 if dv < 15 else 4
            intensity_matrix[i, idx_int] += 1
            idx_type = 0 if dv < 1.2 else 1 if dv < 4.0 else 2 if dv < 15.0 else 3
            type_matrix[i, idx_type] += 1

    conn.close()
    return labels[::-1], type_matrix[::-1], intensity_matrix[::-1], dv_sums[::-1], obs_raw[::-1]


# --- 3. UI SETUP ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 11), facecolor='#0d1117')
plt.subplots_adjust(top=0.82, bottom=0.38, wspace=0.4, left=0.08, right=0.82)

# UPDATED TITLE
fig.suptitle("CHINESE GEO ASSET MANEUVER ANALYTICS - A LOOK IN THE PAST", color='white', fontsize=22, fontweight='bold',
             y=0.97)
scope_note = fig.text(0.45, 0.90, "", ha='center', color='#8b949e', fontsize=12, fontweight='bold')

# UNIFORM DATA SCIENCE NOTES
fig.text(0.84, 0.85, "DATA SCIENCE NOTES\n" + "-" * 20, color='#58a6ff', fontsize=12, fontweight='bold')

fig.text(0.84, 0.72,
         "DATA CLEANING:\nPublic TLE data often\ncontains 'jitter' or small\nerrors. This used median\nfiltering to smooth \nnoise and reduce (prevent) false\nmaneuver detections.",
         color='#8b949e', fontsize=10)

fig.text(0.84, 0.55,
         "GVE DEFINED:\nGauss Variational Equations\n(GVE) is the approach used to\nturn orbital changes into\npropulsion estimates.\nThey calculate how a 'push'\nin one direction alters the\nsatellite's overall path.",
         color='#8b949e', fontsize=10)

fig.text(0.84, 0.38,
         "PROPAGATOR SELECTION:\nSGP4 is used because public\nTLEs are specifically built\nfor it. Using SGP8 would\ncreate math errors due to\nmismatched data sources,\nfitting algorithms, and\nanalytical consistency.",
         color='#8b949e', fontsize=10)

# --- 4. TIMELINE & SOURCE ---
ax_time = plt.axes([0.15, 0.12, 0.6, 0.01], facecolor='#161b22')
ax_time.set_xlim(START_DATE, END_DATE)
ax_time.tick_params(axis='x', colors='#8b949e', labelsize=8)
ax_time.get_yaxis().set_visible(False)

for ev in SIGNIFICANT_EVENTS:
    point, = ax_time.plot(ev['date'], 0, 'D', color='#f85149', markersize=8, picker=True, pickradius=10, clip_on=False)
    point.set_gid((ev['date'] - START_DATE).days)
    point.set_label(f"{ev['label']}: {ev['detail']}")

fig.text(0.45, 0.02, "SOURCE: celestrak.org | TLE SOURCE: 18th SDS | UPDATED: JAN 2026", ha='center', color='gray',
         fontsize=8, family='monospace')


# --- 5. UPDATE LOGIC & CONTROLS ---
def update(val):
    global SENSITIVITY_MODE, CI_OVERLAY_ACTIVE
    current_dt = START_DATE + timedelta(days=int(val))
    labels, type_data, int_data, dv_sums, obs_counts = fetch_operational_data(current_dt.strftime('%Y-%m-%d'),
                                                                              SENSITIVITY_MODE)

    for ax in [ax1, ax2, ax3]:
        ax.clear();
        ax.set_facecolor('#161b22')
        if CI_OVERLAY_ACTIVE:
            for i, n in enumerate(obs_counts):
                if n < 50: ax.axhspan(i - 0.4, i + 0.4, color='#f85149', alpha=0.15, hatch='///')

    for i in range(len(labels)):
        for j in range(4):
            if type_data[i, j] > 0:
                ax1.scatter(j, i, s=np.sqrt(type_data[i, j]) * 320, color='#58a6ff', alpha=0.6, edgecolors='white')
                ax1.text(j, i, int(type_data[i, j]), color='white', ha='center', va='center', fontsize=9,
                         fontweight='bold')
    ax1.set_xticks(range(4));
    ax1.set_xticklabels(M_CLASSES, color='#8b949e', rotation=25, ha='right')
    ax1.set_yticks(range(len(labels)));
    ax1.set_yticklabels(labels, color='white', fontsize=9)
    ax1.set_title("MANEUVER COUNT", color='white', pad=20)

    bottom = np.zeros(len(labels))
    for b in range(5):
        ax2.barh(labels, int_data[:, b], left=bottom, color=INTENSITY_COLORS[b], label=INTENSITY_BINS[b])
        bottom += int_data[:, b]
    if bottom.max() > 0: ax2.set_xlim(0, bottom.max() * 1.3)
    for i, total in enumerate(bottom):
        ax2.text(total + (ax2.get_xlim()[1] * 0.02), i, f"Sum: {int(total)}", color='white', va='center', fontsize=9)
    ax2.set_yticklabels([]);
    ax2.set_title("INTENSITY BY BIN", color='white', pad=20)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, labelcolor='white')

    ax3.barh(labels, dv_sums, color='#f85149', alpha=0.8);
    ax3.set_yticklabels([])
    if len(dv_sums) > 0 and max(dv_sums) > 0: ax3.set_xlim(0, max(dv_sums) * 1.3)
    ax3.set_title("TOTAL MISSION Delta-V", color='white', pad=20)
    for i, v in enumerate(dv_sums):
        ax3.text(v + (ax3.get_xlim()[1] * 0.03), i, f"{int(v)}", color='white', va='center', fontweight='bold')

    scope_note.set_text(f"AUDIT WINDOW: 01 JAN 2020 - {current_dt.strftime('%d %b %Y').upper()}")
    fig.canvas.draw_idle()


slider_ax = plt.axes([0.25, 0.18, 0.4, 0.02], facecolor='#30363d')
slider = Slider(slider_ax, 'Timeline', 0, TOTAL_DAYS, valinit=TOTAL_DAYS, valstep=1)
slider.on_changed(update)

btn_prev = Button(plt.axes([0.2, 0.18, 0.03, 0.02], facecolor='#21262d'), '<', color='white')
btn_next = Button(plt.axes([0.67, 0.18, 0.03, 0.02], facecolor='#21262d'), '>', color='white')

btn_sens = Button(plt.axes([0.12, 0.26, 0.08, 0.03], facecolor='#238636'), f"MODE: {SENSITIVITY_MODE}", color='white')
fig.text(0.205, 0.27, "(Standard: Chem burns | Sensitive: Electric/Ion drift)", color='#8b949e', fontsize=9,
         style='italic')

btn_ci = Button(plt.axes([0.45, 0.26, 0.14, 0.03], facecolor='#30363d'), "SAMPLE CONFIDENCE OVERLAY", color='white')
fig.text(0.595, 0.27, "(N < 50 indicates maneuver uncertainty)", color='#8b949e', fontsize=9, style='italic')


def toggle_sens(e):
    global SENSITIVITY_MODE
    SENSITIVITY_MODE = "SENSITIVE" if SENSITIVITY_MODE == "STANDARD" else "STANDARD"
    btn_sens.label.set_text(f"MODE: {SENSITIVITY_MODE}");
    update(slider.val)


def toggle_ci(e):
    global CI_OVERLAY_ACTIVE
    CI_OVERLAY_ACTIVE = not CI_OVERLAY_ACTIVE
    btn_ci.color = '#1f6feb' if CI_OVERLAY_ACTIVE else '#30363d';
    update(slider.val)


btn_prev.on_clicked(lambda e: slider.set_val(max(0, slider.val - 30)))
btn_next.on_clicked(lambda e: slider.set_val(min(TOTAL_DAYS, slider.val + 30)))
btn_sens.on_clicked(toggle_sens);
btn_ci.on_clicked(toggle_ci)

update(TOTAL_DAYS);
plt.show()