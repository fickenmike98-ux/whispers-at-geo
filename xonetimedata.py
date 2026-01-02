import sqlite3
import requests
import pandas as pd

DB_PATH = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"
# Expanded list for a more complete dashboard
FLEET_IDS = {55222: 'SJ-23', 49330: 'SJ-21', 41838: 'SJ-17', 57091: 'SJ-25', 40093: 'GSSAP 1'}


def ingest_high_res_history():
    conn = sqlite3.connect(DB_PATH)
    print("Connecting to CelesTrak API...")

    for norad_id, name in FLEET_IDS.items():
        # Using the 'FORMAT=JSON' endpoint which often provides more historical context
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=JSON"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)

                # Standardizing columns
                # Note: CelesTrak uses 'NORAD_CAT_ID', 'EPOCH', 'MEAN_MOTION'
                subset = pd.DataFrame()
                subset['norad_id'] = df['NORAD_CAT_ID']
                subset['epoch'] = df['EPOCH']
                subset['mean_motion'] = df['MEAN_MOTION']

                # We use a temporary table to avoid the IntegrityError
                subset.to_sql('temp_ingest', conn, if_exists='replace', index=False)

                # SQL "INSERT OR IGNORE" handles the UNIQUE constraint perfectly
                conn.execute("""
                             INSERT
                             OR IGNORE INTO gp_history (norad_id, epoch, mean_motion)
                             SELECT norad_id, epoch, mean_motion
                             FROM temp_ingest
                             """)
                conn.commit()

                # Check how many records we now have for this satellite
                count = conn.execute(f"SELECT COUNT(*) FROM gp_history WHERE norad_id={norad_id}").fetchone()[0]
                print(f"Total historical points for {name}: {count}")

        except Exception as e:
            print(f"Error processing {name}: {e}")

    conn.close()
    print("\nIngest complete. Check your dashboard at Index 71/72.")


ingest_high_res_history()