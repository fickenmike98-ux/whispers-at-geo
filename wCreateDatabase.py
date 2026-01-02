import sqlite3


def init_safe_db():
    # We rename the DB to reflect the public source
    conn = sqlite3.connect("celestrak_public_sda.db")
    cursor = conn.cursor()

    # Table optimized for CelesTrak JSON OMM format
    # Using the primary key (norad_id, epoch) prevents duplicate training data
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS gp_history
                   (
                       norad_id
                       INTEGER,
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
                       arg_perigee
                       REAL,
                       mean_anomaly
                       REAL,
                       bstar
                       REAL,
                       mean_element_theory
                       TEXT,
                       PRIMARY
                       KEY
                   (
                       norad_id,
                       epoch
                   )
                       )
                   """)

    # Optional: Table for your "Public-Safe" Strategic Overrides
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS public_strategic_context
                   (
                       date
                       TEXT
                       PRIMARY
                       KEY,
                       vix_index
                       REAL,
                       countdown_2027
                       REAL,
                       tension_proxy
                       REAL
                   )
                   """)

    conn.commit()
    conn.close()
    print("[+] Public-safe database initialized.")


if __name__ == "__main__":
    init_safe_db()