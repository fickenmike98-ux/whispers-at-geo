import sqlite3
import os

# Updated to your specific DB name
db_path = r"C:\Users\MDesktop\PycharmProjects\PythonProject1\celestrak_public_sda.db"


def audit_celestrak_db(path):
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        return

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    print("=" * 50)
    print(f"AUDITING: {os.path.basename(path)}")
    print("=" * 50)

    # 1. List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        print("No tables found in this database.")
        return

    for t_name in [t[0] for t in tables]:
        print(f"\nTABLE: {t_name}")

        # 2. Get Column Info
        cursor.execute(f"PRAGMA table_info({t_name});")
        cols = cursor.fetchall()
        print(f"{'ID':<3} | {'Name':<20} | {'Type':<10}")
        print("-" * 40)
        for col in cols:
            print(f"{col[0]:<3} | {col[1]:<20} | {col[2]:<10}")

        # 3. Peek at the data format
        try:
            cursor.execute(f"SELECT * FROM {t_name} LIMIT 1;")
            row = cursor.fetchone()
            print(f"\nSAMPLE RECORD FROM {t_name}:")
            print(row)
        except Exception as e:
            print(f"Could not read sample: {e}")

        print("\n" + "=" * 30)

    conn.close()


audit_celestrak_db(db_path)