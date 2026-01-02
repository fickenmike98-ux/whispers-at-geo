import pandas as pd

# The 'Trigger' list (sample of the 100 discrete events)
triggers = [
    ('2020-08-10', 0.65), ('2020-09-17', 0.70), ('2021-10-04', 0.85),
    ('2022-08-02', 1.00), ('2022-12-25', 0.80), ('2023-04-08', 0.90),
    ('2024-05-23', 0.95), ('2024-10-14', 0.98), ('2025-11-07', 0.90),
    ('2025-12-18', 0.95), ('2025-12-29', 1.00)  # Justice Mission 2025
    # ... + 90 more discrete dates
]


def generate_tension_tensor(start_date='2020-01-01', end_date='2026-01-01'):
    # Create daily timeline
    timeline = pd.date_range(start=start_date, end=end_date, freq='D')
    tension_df = pd.DataFrame(index=timeline)
    tension_df['score'] = 0.2  # Baseline tension

    # Apply the 100 discrete pulses
    for date, score in triggers:
        tension_df.loc[date:, 'score'] = score

    # Optional: Apply "Cooling" (linear decay of 0.01 per week if no events happen)
    # This simulates how people/markets/satellites relax after a drill ends.
    return tension_df


# Merge this into your SQLite satellite data
tension_tensor = generate_tension_tensor()