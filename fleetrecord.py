# --- 1. CONFIG & MAPPING (Using NORAD IDs for accuracy) ---
DB_PATH = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"

# Mapping Names to NORAD IDs found in CelesTrak
CN_FLEET = {
    'SJ-17': 41838, 'SJ-20': 44909, 'SJ-21': 49330,
    'SJ-23': 55222, 'SJ-25': 57091, 'TJS-12': 59858
}
US_FLEET = {
    'GSSAP 1': 40093, 'GSSAP 2': 40094, 'GSSAP 3': 41732,
    'GSSAP 4': 41733, 'GSSAP 5': 50981, 'GSSAP 6': 50982
}


# --- 2. THE HISTORICAL ENGINE ---
def fetch_real_data(id_map, end_date):
    conn = sqlite3.connect(DB_PATH)
    fleet_names = list(id_map.keys())
    results = np.zeros((len(fleet_names), len(INTENSITY_BINS)))

    try:
        for i, (name, norad_id) in enumerate(id_map.items()):
            # QUERYING gp_history INSTEAD OF raw_elements
            query = f"""
                SELECT epoch, mean_motion 
                FROM gp_history 
                WHERE norad_id = {norad_id} 
                AND epoch <= '{end_date}'
                ORDER BY epoch ASC
            """
            df = pd.read_sql_query(query, conn)

            if len(df) < 2:
                # If no history is found, we'll try a very small random jitter
                # just to keep the UI from being a total blank during setup
                continue

            print(f"Analyzing {len(df)} historical records for {name}...")

            # Calculate Deltas
            df['dn'] = df['mean_motion'].diff().abs()

            # THE KEY TO MORE RECORDS:
            # We use a very low threshold (0.00001) to catch station-keeping
            maneuvers = df[df['dn'] > 0.00001].copy()
            maneuvers['dv_est'] = maneuvers['dn'] * 70000

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
        print(f"History Engine Error: {e}")
    finally:
        conn.close()
    return results

# --- 3. UI UPDATE (Adjusting the Call) ---
# Change your update(val) function to pass the dictionaries:
# m_cn = fetch_real_data(CN_FLEET, date_str)
# m_us = fetch_real_data(US_FLEET, date_str)