import numpy as np
import pandas as pd
from scipy.signal import correlate


def analyze_lead_lag(sat_name, orbital_deltas, tension_scores):
    # Cross-correlate normalized signals
    corr = correlate(orbital_deltas - np.mean(orbital_deltas),
                     tension_scores - np.mean(tension_scores))

    lags = np.arange(-len(orbital_deltas) + 1, len(orbital_deltas))
    best_lag = lags[np.argmax(corr)]

    if best_lag < 0:
        print(f"[!] {sat_name} LEADS geopolitics by {abs(best_lag)} days.")
    elif best_lag > 0:
        print(f"[*] {sat_name} LAGS geopolitics by {best_lag} days.")
    else:
        print(f"[-] {sat_name} maneuvers are synchronized with events.")

# Example Usage:
# analyze_lead_lag("SJ-21", df['dn'].values, df['tension'].values)