import sqlite3
import pandas as pd

# --- CONFIG ---
DB_PATH = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"

# The "Heavy Hitters" we want to verify
HEAVY_HITTERS = {
    55222: "SJ-23 (Inspector)",
    49330: "SJ-21 (Tug)",
    41838: "SJ-17 (Survey)",
    46610: "GAOFEN-13 (Baseline)",
    58957: "TJS-11 (SIGINT)",
    40892: "SBSS 1 (US SDA)",
    41194: "NAVSTAR GPS (US PNT)"
}


def run_census():
    conn = sqlite3.connect(DB_PATH)

    # Get summary of all data
    query = """
            SELECT norad_id, COUNT(*) as record_count, MIN(epoch) as start_date, MAX(epoch) as end_date
            FROM gp_history
            GROUP BY norad_id
            ORDER BY record_count DESC \
            """
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("\n" + "=" * 80)
    print("--- SDA DATABASE CENSUS REPORT ---")
    print("=" * 80)

    # 1. Check Heavy Hitters specifically
    print(f"{'STATUS':<10} | {'NAME':<20} | {'SCN':<8} | {'RECORDS':<8} | {'COVERAGE RANGE'}")
    print("-" * 80)

    found_ids = df['norad_id'].tolist()

    for scn, name in HEAVY_HITTERS.items():
        if scn in found_ids:
            row = df[df['norad_id'] == scn].iloc[0]
            status = "✅ LOADED"
            details = f"{int(row['record_count']):<8} | {row['start_date'][:10]} to {row['end_date'][:10]}"
        else:
            status = "❌ MISSING"
            details = f"{'0':<8} | No Data Found"

        print(f"{status:<10} | {name:<20} | {scn:<8} | {details}")

    print("\n" + "=" * 80)
    print("--- TOP 10 OTHER ASSETS IN DATABASE ---")
    print("=" * 80)

    # 2. Show other top assets not in heavy hitters
    others = df[~df['norad_id'].isin(HEAVY_HITTERS.keys())].head(10)
    for _, row in others.iterrows():
        print(
            f"SCN: {int(row['norad_id']):<8} | Records: {int(row['record_count']):<8} | Range: {row['start_date'][:10]} to {row['end_date'][:10]}")

    print("\n" + "=" * 80)
    print(f"TOTAL UNIQUE SATELLITES: {len(df)}")
    print(f"TOTAL SYSTEM RECORDS: {df['record_count'].sum():,}")
    print("=" * 80)


if __name__ == "__main__":
    run_census()