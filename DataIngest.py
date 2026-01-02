import sqlite3
import requests
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
DB_NAME = "celestrak_public_sda.db"
# High-Interest Assets for targeted sync
HIGH_PROFILE_IDS = [49330, 41838, 44207, 45155, 50466, 59728]
TARGET_KEYWORDS = ["SJ-", "TJS", "ZHONGXING", "SHIYAN", "SY-", "GSSAP"]


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS raw_elements
    (
        norad_id
        INTEGER,
        name
        TEXT,
        epoch
        TEXT,
        mean_motion
        REAL,
        eccentricity
        REAL,
        inclination
        REAL,
        raan
        REAL,
        bstar
        REAL,
        UNIQUE
                 (
        norad_id,
        epoch
                 ))''')
    conn.commit()
    conn.close()


def safe_ingest(data_json):
    if not data_json: return 0
    entries = data_json if isinstance(data_json, list) else [data_json]

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    count = 0
    for entry in entries:
        try:
            name = entry.get('OBJECT_NAME', 'UNKNOWN').upper()
            # Ingest if it matches keywords OR is in our high-profile list
            if any(k in name for k in TARGET_KEYWORDS) or entry.get('NORAD_CAT_ID') in HIGH_PROFILE_IDS:
                c.execute("""INSERT
                OR IGNORE INTO raw_elements 
                    (norad_id, name, epoch, mean_motion, eccentricity, inclination, raan, bstar)
                    VALUES (?,?,?,?,?,?,?,?)""", (
                              entry['NORAD_CAT_ID'], name, entry['EPOCH'],
                              entry['MEAN_MOTION'], entry['ECCENTRICITY'], entry['INCLINATION'],
                              entry['RA_OF_ASC_NODE'], entry.get('BSTAR', 0)
                          ))
                count += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return count


def run_sync_cycle():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # 1. GEO Group Sync (The bulk fetch)
    print(f"[*] [{datetime.now().strftime('%H:%M')}] Requesting GEO Batch...")
    try:
        r = requests.get("https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=JSON", headers=headers,
                         timeout=30)
        if r.status_code == 200:
            added = safe_ingest(r.json())
            print(f" [+] Success: Ingested {added} targets from GEO batch.")
    except Exception as e:
        print(f" [!] GEO Batch Error: {e}")

    # 2. Targeted High-Profile Sync (Ensures we don't miss non-GEO assets like SJ-21 in transfer)
    for nid in HIGH_PROFILE_IDS:
        time.sleep(random.uniform(3, 6))  # Stealth delay
        try:
            r = requests.get(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={nid}&FORMAT=JSON", headers=headers,
                             timeout=20)
            if r.status_code == 200:
                safe_ingest(r.json())
        except Exception as e:
            print(f" [!] Targeted ID {nid} failed: {e}")


# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    init_db()
    print("=== SDA PERSISTENCE INGEST STARTED ===")
    print("[*] Note: To backfill history, let this run for 24-48 hours.")

    try:
        while True:
            run_sync_cycle()

            # Wait 4 hours (14400 seconds) + jitter to avoid detection
            wait_time = 14400 + random.randint(0, 300)
            next_run = wait_time / 3600
            print(f"[*] Cycle complete. Next sync in ~{next_run:.1f} hours...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n[!] Ingest stopped by user. Database preserved.")