import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Location": [
        "Guillermo Mendoza Building",
        "HWPL Monument",
        "Information Office",
        "Administrative Building",
        "Auditorium",
        "CIT Building",
        "Near Gate 6",
        "Near Gate 7",
        "Sports Center",
        "Pascual T, Galura Street"
    ],
    "PM1.0 (µg/m³)": [8, 27, 14.9, 14.7, 15, 28.92, 27.21, 20.41, 20.4, 10],
    "PM2.5 (µg/m³)": [12, 37, 18.4, 19.7, 21, 43.04, 40.02, 29.79, 27, 14],
    "PM10 (µg/m³)": [14, 44.14, 19.6, 22.6, 25, 53.94, 49.33, 34.33, 31.2, 16]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10,6))

plt.plot(df["Location"], df["PM1.0 (µg/m³)"], marker='o', label="PM1.0")
plt.plot(df["Location"], df["PM2.5 (µg/m³)"], marker='o', label="PM2.5")
plt.plot(df["Location"], df["PM10 (µg/m³)"], marker='o', label="PM10")

plt.title("Particulate Matter Concentration by Location", fontsize=14)
plt.xlabel("Location", fontsize=12)
plt.ylabel("Concentration (µg/m³)", fontsize=12)
plt.xticks(rotation=75)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()