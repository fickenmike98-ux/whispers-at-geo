import sqlite3

conn = sqlite3.connect("celestrak_sda.db")
cursor = conn.cursor()

# 1. Let's see exactly what columns are in the table (Debugging)
print("[*] Table Structure:")
cursor.execute("PRAGMA table_info(raw_elements)")
for col in cursor.fetchall():
    print(f" - {col[1]} ({col[2]})")

print("\n" + "="*35)
print(f"{'Satellite Name':<20} | {'Record Count':<10}")
print("-" * 35)

# 2. Updated Query using the correct column 'name'
query = "SELECT name, COUNT(*) FROM raw_elements GROUP BY name"
try:
    for row in cursor.execute(query):
        print(f"{row[0]:<20} | {row[1]:<10}")
except Exception as e:
    print(f"[!] Error: {e}")

conn.close()