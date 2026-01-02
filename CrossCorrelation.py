import numpy as np
import pandas as pd
from scipy.signal import correlate


def calculate_strategic_lead(sat_id, maneuver_series, tension_series):
    # Normalize signals
    maneuver_norm = (maneuver_series - np.mean(maneuver_series)) / (np.std(maneuver_series) + 1e-9)
    tension_norm = (tension_series - np.mean(tension_series)) / (np.std(tension_series) + 1e-9)

    # Calculate cross-correlation for a 14-day window
    correlation = correlate(maneuver_norm, tension_norm, mode='full')
    lags = np.arange(-len(maneuver_norm) + 1, len(maneuver_norm))

    # Focus on the 'Strategic Lead' window (Maneuvers occurring 0-10 days BEFORE event)
    strategic_window = (lags >= -10) & (lags <= 0)
    best_lag = lags[strategic_window][np.argmax(correlation[strategic_window])]

    return abs(best_lag)


print("[*] Lead-Lag engine configured for 14-day strategic windows.")