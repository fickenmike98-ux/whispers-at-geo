import sqlite3
import pandas as pd

DB_NAME = "celestrak_public_sda.db"


def run_sanity_check():
    conn = sqlite3.connect(DB_NAME)

    # 1. Check Total Row Counts
    total_rows = pd.read_sql_query("SELECT COUNT(*) as count FROM raw_elements", conn).iloc[0]['count']
    print(f"[*] Total Records in Database: {total_rows}")

    if total_rows == 0:
        print("[!] DATABASE IS EMPTY. Run DataIngest.py first.")
        return

    # 2. Check Unique Assets
    unique_assets = pd.read_sql_query("SELECT DISTINCT name, norad_id FROM raw_elements", conn)
    print(f"[*] Unique High-Interest Assets Found: {len(unique_assets)}")
    print(unique_assets.head(10))

    # 3. Physics Verification: Mean Motion Delta for a specific target (e.g., SJ-21)
    target_id = 49330
    print(f"\n[*] Analyzing Physics for NORAD ID {target_id} (SJ-21)...")

    query = f"""
        SELECT epoch, mean_motion, inclination 
        FROM raw_elements 
        WHERE norad_id = {target_id} 
        ORDER BY epoch DESC LIMIT 5
    """
    df = pd.read_sql_query(query, conn)

    if not df.empty:
        print(df)
        # Calculate Delta
        if len(df) > 1:
            delta = abs(df['mean_motion'].iloc[0] - df['mean_motion'].iloc[1])
            print(f"\n[+] Current Mean Motion Delta: {delta:.8f}")
            if delta > 0.0001:
                print(" [!] ALERT: Significant orbital change detected in last two epochs.")
            else:
                print(" [OK] Orbit appears stable (within normal perturbation bounds).")
    else:
        print(f"[!] No records found for {target_id}. Check if ingest was filtered.")

    conn.close()


if __name__ == "__main__":
    run_sanity_check()