import os
import sqlite3

def sanitize_environment():
    db_file = 'sda_local.db'
    model_file = 'sda_local_trained_v1.h5'

    print("[*] Starting sanitation process...")

    # 1. Close connections and delete the database
    if os.path.exists(db_file):
        try:
            # Ensure no handles are open
            conn = sqlite3.connect(db_file)
            conn.close()
            os.remove(db_file)
            print(f"[+] Successfully deleted {db_file}")
        except Exception as e:
            print(f"[!] Error deleting database: {e}")
    else:
        print("[?] Database file not found. Already clean.")

    # 2. Delete the trained model (contains derived insights from restricted data)
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"[+] Successfully deleted {model_file}")

    print("[*] Sanitation complete. Environment is safe for non-ODR work.")

if __name__ == "__main__":
    sanitize_environment()