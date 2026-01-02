import sqlite3
import pandas as pd
import glob
import os

CSV_FOLDER = r"C:\Users\MDesktop\Desktop\CSVs"
DB_PATH = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"


def ingest_csv_robust():
    conn = sqlite3.connect(DB_PATH)
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    print(f"Found {len(csv_files)} files. Starting Robust Ingestion...")

    # Expanded Alias Lists
    id_aliases = ['NORAD_CAT_ID', 'OBJECT_ID', 'CATNR', 'SATNR', 'SATELLITE_NUMBER', 'NORAD_ID']
    epoch_aliases = ['EPOCH', 'DATE', 'TIME', 'UTC_EPOCH', 'CREATION_DATE']
    mm_aliases = ['MEAN_MOTION', 'MM', 'MOTION', 'N_REVS_DAY', 'MEAN_MOTION_DOT']

    for file in csv_files:
        try:
            # Try reading with different delimiters just in case (comma, then space)
            try:
                df = pd.read_csv(file)
            except:
                df = pd.read_csv(file, sep='\s+')

            df.columns = [c.upper().strip() for c in df.columns]

            # Smart Matching
            target_id = next((c for c in id_aliases if c in df.columns), None)
            target_epoch = next((c for c in epoch_aliases if c in df.columns), None)
            target_mm = next((c for c in mm_aliases if c in df.columns), None)

            if target_id and target_epoch and target_mm:
                subset = pd.DataFrame()
                subset['norad_id'] = pd.to_numeric(df[target_id], errors='coerce').fillna(0).astype(int)
                subset['epoch'] = df[target_epoch]
                subset['mean_motion'] = pd.to_numeric(df[target_mm], errors='coerce')

                # Cleanup
                subset = subset[subset['norad_id'] > 0].dropna()

                subset.to_sql('temp_batch', conn, if_exists='replace', index=False)
                conn.execute("""
                             INSERT
                             OR IGNORE INTO gp_history (norad_id, epoch, mean_motion)
                             SELECT norad_id, epoch, mean_motion
                             FROM temp_batch
                             """)
                conn.commit()
                print(f"✅ SUCCESS: {os.path.basename(file)} | {len(subset)} rows added.")
            else:
                print(f"❌ FAIL: {os.path.basename(file)} | Missing columns. Headers were: {list(df.columns)}")

        except Exception as e:
            print(f"⚠️ ERROR: {os.path.basename(file)} | {e}")

    total = conn.execute("SELECT COUNT(*) FROM gp_history").fetchone()[0]
    print(f"\n--- INGESTION COMPLETE ---")
    print(f"Total Records in Database: {total:,}")
    conn.close()


if __name__ == "__main__":
    ingest_csv_robust()