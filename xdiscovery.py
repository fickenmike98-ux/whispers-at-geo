import requests


def discover_targets():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=JSON"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=60)
        data = response.json()

        print(f"{'NORAD ID':<10} | {'Object Name':<25}")
        print("-" * 40)

        # Keywords to look for
        keywords = ["SJ", "TJS", "GSSAP", "ZHONGXING", "CHINASAT", "WGS"]

        for entry in data:
            name = entry['OBJECT_NAME'].upper()
            if any(key in name for key in keywords):
                print(f"{entry['NORAD_CAT_ID']:<10} | {entry['OBJECT_NAME']:<25}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    discover_targets()