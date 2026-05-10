# =========================
# Carbon Signal Validation 
# =========================

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 0. Settings
# -------------------------
FILE_PATH = r"C:\Users\LIUWENLONG\Desktop\carbon-aware-dr-wenlong\data\PT_2024_hourly.csv"
TIME_COL = "Datetime (UTC)"
CI_COL = "Carbon intensity gCO2/kWh (direct)"
TIMEZONE = "Europe/Lisbon"

# -------------------------
# 1. Load data
# -------------------------
df = pd.read_csv(FILE_PATH)

# Convert time column to datetime and set timezone
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
df = df.dropna(subset=[TIME_COL, CI_COL]).copy()

# Convert to local timezone
df[TIME_COL] = df[TIME_COL].dt.tz_convert(TIMEZONE)

# Convert carbon intensity to numeric
df[CI_COL] = pd.to_numeric(df[CI_COL], errors="coerce")
df = df.dropna(subset=[CI_COL]).copy()

# -------------------------
# 2. Feature engineering
# -------------------------
df["date"] = df[TIME_COL].dt.date
df["hour"] = df[TIME_COL].dt.hour
df["month"] = df[TIME_COL].dt.month

def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["season"] = df["month"].apply(month_to_season)

# -------------------------
# 3. Plot 1 - time series
# -------------------------
plt.figure(figsize=(12, 5))
plt.plot(df[TIME_COL], df[CI_COL])
plt.xlabel("Time")
plt.ylabel("Carbon intensity (gCO2/kWh)")
plt.title("Hourly Carbon Intensity (Portugal)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -------------------------
# 4. Plot 2 - daily variation
# -------------------------
hourly = df.groupby("hour")[CI_COL].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(hourly["hour"], hourly[CI_COL], marker="o")
plt.xlabel("Hour")
plt.ylabel("Avg carbon intensity (gCO2/kWh)")
plt.title("Average Intraday Carbon Intensity")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 5. Plot 3 - monthly variation
# -------------------------
monthly = df.groupby("month")[CI_COL].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(monthly["month"], monthly[CI_COL], marker="o")
plt.xlabel("Month")
plt.ylabel("Avg carbon intensity (gCO2/kWh)")
plt.title("Monthly Carbon Intensity")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 6. Plot 4 - seasonal variation
# -------------------------
season_order = ["Winter", "Spring", "Summer", "Autumn"]

seasonal = df.groupby("season")[CI_COL].mean().reindex(season_order).reset_index()

plt.figure(figsize=(8, 5))
plt.bar(seasonal["season"], seasonal[CI_COL])
plt.xlabel("Season")
plt.ylabel("Avg carbon intensity (gCO2/kWh)")
plt.title("Seasonal Carbon Intensity")
plt.tight_layout()
plt.show()

# -------------------------
# 7. Plot 5 - season & hour heatmap
# -------------------------
season_hour = df.groupby(["season", "hour"])[CI_COL].mean().reset_index()

plt.figure(figsize=(10, 6))
for s in season_order:
    temp = season_hour[season_hour["season"] == s]
    plt.plot(temp["hour"], temp[CI_COL], marker="o", label=s)

plt.xlabel("Hour")
plt.ylabel("Carbon intensity (gCO2/kWh)")
plt.title("Intraday Carbon Intensity by Season")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 8. Plot 6 - Monthly boxplot
# -------------------------
plt.figure(figsize=(10, 6))

df.boxplot(column=CI_COL, by="month")

plt.xlabel("Month")
plt.ylabel("Carbon intensity (gCO2/kWh)")
plt.title("Monthly Distribution of Carbon Intensity")
plt.suptitle("")  
plt.tight_layout()
plt.show()