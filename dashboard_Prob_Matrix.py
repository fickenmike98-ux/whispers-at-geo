import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SATELLITE FLEET TO MONITOR
TARGET_SATS = ['SJ-17', 'SJ-20', 'SJ-21', 'SJ-23', 'SJ-25', 'TJS-12']


def generate_forecast_dashboard(model_results):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Probabilities for [24h, 72h, 1wk]
    horizons = ['24 Hours', '72 Hours', '1 Week']

    # Colors for the 'Alert Level'
    # Red: >70%, Orange: 40-70%, Green: <40%
    im = ax.imshow(model_results, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(horizons)))
    ax.set_yticks(np.arange(len(TARGET_SATS)))
    ax.set_xticklabels(horizons, fontweight='bold')
    ax.set_yticklabels(TARGET_SATS, fontweight='bold')

    # Add text annotations
    for i in range(len(TARGET_SATS)):
        for j in range(len(horizons)):
            val = model_results[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight='bold')

    plt.title(f"EXPERIMENTAL FLEET MANEUVER PROBABILITY: {pd.Timestamp.now().strftime('%Y-%m-%d')}", fontsize=14)
    plt.colorbar(im, label='Confidence (%)')
    plt.tight_layout()
    plt.show()


# Example mock data for the dashboard logic:
# Rows: Sats, Cols: [24h, 72h, 1wk]
mock_inference = np.array([
    [12.5, 45.2, 88.1],  # SJ-17
    [5.1, 8.4, 12.0],  # SJ-20
    [65.4, 92.1, 98.5],  # SJ-21 (High Alert)
    [2.3, 5.1, 15.2],  # SJ-23
    [41.0, 55.0, 72.0],  # SJ-25
    [8.0, 12.0, 35.0]  # TJS-12
])

# generate_forecast_dashboard(mock_inference)