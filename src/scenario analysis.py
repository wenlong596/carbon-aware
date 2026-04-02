import pandas as pd
import numpy as np

# =========================
# Step 1 - Load & Basic Prep
# =========================

 
#CSV_PATH = r"C:\Users\LIUWENLONG\Desktop\carbon-aware-dr-wenlong\data\raw\SCML-RuadaVinhaFirst_FirstFloor.csv"
CSV_PATH = r"C:\Users\LIUWENLONG\Desktop\carbon-aware-dr-wenlong\data\raw\SCML-RainhaSanta_SecondFloor.csv"


TIME_COL  = "timestamp"
E_COL     = "energy_kWh"
P_COL     = "wholesale_price_EUR_per_kWh"
CI_KG_COL = "carbon_intensity_kgCO2_per_kWh"

df = pd.read_csv(CSV_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce").dt.tz_convert("Europe/Lisbon")
df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

# 基本列检查
need_cols = [TIME_COL, E_COL, P_COL, CI_KG_COL]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Your columns are: {list(df.columns)}")

df["date"] = df[TIME_COL].dt.date
df["dow"] = df[TIME_COL].dt.dayofweek  # Mon=0 ... Sun=6
df["minute_of_day"] = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute

# 数值化
df[E_COL] = pd.to_numeric(df[E_COL], errors="coerce")
df[P_COL] = pd.to_numeric(df[P_COL], errors="coerce")
df[CI_KG_COL] = pd.to_numeric(df[CI_KG_COL], errors="coerce")


# =========================
# Step 1A - TOU (Portugal 3-period) ERES
# =========================

# Periods: offpeak / shoulder / peak
def tou_portugal_3period(dow: int, minute_of_day: int) -> str:
    # Sunday: offpeak all day
    if dow == 6:
        return "offpeak"

    # Saturday: peak 09:30-13:00, otherwise offpeak 
    if dow == 5:
        peak_start = 9 * 60 + 30
        peak_end   = 13 * 60
        if peak_start <= minute_of_day < peak_end:
            return "peak"
        return "offpeak"

    # Weekdays (Mon-Fri)
    # offpeak: 22:00-24:00 and 00:00-08:00
    if minute_of_day < 8 * 60 or minute_of_day >= 22 * 60:
        return "offpeak"

    # peak: 09:30-12:30
    peak_start = 9 * 60 + 30
    peak_end   = 12 * 60 + 30
    if peak_start <= minute_of_day < peak_end:
        return "peak"

    # shoulder: rest of 08:00-22:00 excluding peak
    return "shoulder"

df["tou_period"] = [
    tou_portugal_3period(d, m)
    for d, m in zip(df["dow"].to_numpy(), df["minute_of_day"].to_numpy())
]


# =========================
# Step 1B - Electricity Tariffs (Flat / TOU(3) / RTP)
# =========================

# RTP: wholesale proxy
df["price_rtp"] = df[P_COL]

# Flat: daily mean wholesale
df["price_flat"] = df.groupby("date")[P_COL].transform("mean")

# TOU(3): per-day period means
daily_mean = df.groupby("date")[P_COL].mean()

tou_means = (
    df.groupby(["date", "tou_period"])[P_COL]
      .mean()
      .unstack("tou_period")
      .rename(columns={"offpeak": "P_off", "shoulder": "P_shoulder", "peak": "P_peak"})
)

tou_means["P_day"] = daily_mean

# fallback if a period is missing that day
for c in ["P_off", "P_shoulder", "P_peak"]:
    if c not in tou_means.columns:
        tou_means[c] = np.nan
    tou_means[c] = tou_means[c].fillna(tou_means["P_day"])

df = df.merge(tou_means[["P_off", "P_shoulder", "P_peak"]], on="date", how="left")

# assign hourly TOU price
df["price_tou"] = np.select(
    [
        df["tou_period"].eq("offpeak"),
        df["tou_period"].eq("shoulder"),
        df["tou_period"].eq("peak"),
    ],
    [
        df["P_off"],
        df["P_shoulder"],
        df["P_peak"],
    ],
    default=df["price_flat"]
)


# =========================
# Step 1C - Carbon Valuation (Shadow Prices)
# =========================

CARBON_PRICES = [50, 100, 200]  # €/tCO2

# kgCO2/kWh -> tCO2/kWh
df["tCO2_per_kWh"] = df[CI_KG_COL] / 1000.0

# 生成每个碳价下的 “carbon_cost_EUR” 
for cp in CARBON_PRICES:
    df[f"carbon_price_EUR_per_tCO2_{cp}"] = cp
    # 碳成本(€) = energy(kWh) * tCO2/kWh * €/tCO2
    df[f"carbon_cost_EUR_{cp}"] = df[E_COL] * df["tCO2_per_kWh"] * cp

# “每kWh的碳成本信号”（€/kWh）
for cp in CARBON_PRICES:
    df[f"carbon_cost_EUR_per_kWh_{cp}"] = df["tCO2_per_kWh"] * cp


import os

# =========================
# Step 1D - Output (scenario-ready)
# =========================

out_cols = [
    TIME_COL, "date", "dow", "minute_of_day",
    E_COL, P_COL, CI_KG_COL,
    "tou_period",
    "price_flat", "price_tou", "price_rtp",
    "tCO2_per_kWh",
] + [f"carbon_cost_EUR_{cp}" for cp in CARBON_PRICES] \
  + [f"carbon_cost_EUR_per_kWh_{cp}" for cp in CARBON_PRICES]

out = df[out_cols].copy()

#  获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # 项目根目录

import os

# 获取输入文件名
base_name = os.path.basename(CSV_PATH)
name_no_ext = os.path.splitext(base_name)[0]

# 生成输出文件名
out_file = f"{name_no_ext}_policy_scenarios.csv"

# 拼接路径
OUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    out_file
)

out.to_csv(OUT_PATH, index=False)

print(f"\nSaved: {OUT_PATH}")


# =========================
# Optional sanity check: tariff signal plots
# =========================
import matplotlib.pyplot as plt

# 1) 选一个“完整的一天”来画
daily_counts = df.groupby("date")[TIME_COL].count().sort_values(ascending=False)
valid_days = daily_counts[daily_counts >= 24].index  # 至少2条（1h数据=24条，15min=96条）
if len(valid_days) == 0:
    #随便取第一天
    plot_day = df["date"].iloc[0]
else:
    #plot_day = valid_days[len(valid_days)//2] 中位数那天
    plot_day = np.random.choice(valid_days) #随机选一天
    
day_df = df[df["date"] == plot_day].copy()


# fig 1：同一天 Flat / TOU / RTP 
day_df = day_df.sort_values(TIME_COL).copy()

# 提取 hour
day_df["hour"] = day_df[TIME_COL].dt.hour

# 如果同一小时有重复值，按小时取均值
plot_df = day_df.groupby("hour", as_index=False)[["price_rtp", "price_tou", "price_flat"]].mean()
plot_df = plot_df.sort_values("hour").reset_index(drop=True)


x = plot_df["hour"].tolist()
y_rtp = plot_df["price_rtp"].tolist()
y_tou = plot_df["price_tou"].tolist()
y_flat = plot_df["price_flat"].tolist()

if 0 in plot_df["hour"].values:
    y_rtp.append(plot_df.loc[plot_df["hour"] == 0, "price_rtp"].iloc[0])
    y_tou.append(plot_df.loc[plot_df["hour"] == 0, "price_tou"].iloc[0])
    y_flat.append(plot_df.loc[plot_df["hour"] == 0, "price_flat"].iloc[0])
    x.append(24)

plt.figure(figsize=(9,4.5))

# RTP
plt.plot(
    x,
    y_rtp,
    label="RTP",
    linewidth=2,
)

# TOU
plt.plot(
    x,
    y_tou,
    drawstyle="steps-post",
    label="TOU",
    linewidth=2
)

# Flat
plt.plot(
    x,
    y_flat,
    drawstyle="steps-post",
    label="Flat",
    linewidth=2
)

plt.xlabel("Hour of day")
plt.ylabel("EUR/kWh")
plt.title(f"Tariff signals on {plot_day}")
plt.xticks(range(25))   # 0-24
plt.xlim(0, 24)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# Step 2 — Construct Carbon & Economic Signals
# input: df from Step 1 
# required columns:
#   timestamp
#   energy_kWh
#   carbon_intensity_kgCO2_per_kWh
#   price_flat
#   price_tou
#   price_rtp
#   tou_period
#   date
# =========================

CARBON_PRICES = [50, 100, 200]

def step2_construct_signals(df):
    df2 = df.copy()

    # 1) standardize core signals
    df2["L_kWh"] = pd.to_numeric(df2[E_COL], errors="coerce")
    df2["CI_kgCO2_per_kWh"] = pd.to_numeric(df2[CI_KG_COL], errors="coerce")

    # 2) remove impossible negatives
    df2.loc[df2["L_kWh"] < 0, "L_kWh"] = np.nan
    df2.loc[df2["CI_kgCO2_per_kWh"] < 0, "CI_kgCO2_per_kWh"] = np.nan

    # 3) hourly emissions
    df2["emission_kgCO2"] = df2["L_kWh"] * df2["CI_kgCO2_per_kWh"]
    df2["emission_tCO2"] = df2["emission_kgCO2"] / 1000.0

    # 4) hourly electricity costs under each tariff
    df2["cost_flat"] = df2["L_kWh"] * df2["price_flat"]
    df2["cost_tou"]  = df2["L_kWh"] * df2["price_tou"]
    df2["cost_rtp"]  = df2["L_kWh"] * df2["price_rtp"]

    # 5) optional: shadow carbon values for later evaluation
    for cp in CARBON_PRICES:
       df2[f"carbon_value_{cp}"] = df2["emission_tCO2"] * cp
   
    # 6) make sure date exists
    if "date" not in df2.columns:
        df2["date"] = df2[TIME_COL].dt.date

    return df2


# RUN
df2 = step2_construct_signals(df)

#print("Flat price mean (EUR/kWh) =", df2["price_flat"].mean()) 
#print("TOU period means (EUR/kWh) =") 
#print(df2.groupby("tou_period")["price_tou"].mean()) 

#print("Columns added:", [c for c in df2.columns if c not in df.columns]) 

# Daily summary
daily = df2.groupby("date", as_index=False).agg(
    L_kWh=("L_kWh", "sum"),
    #emission_kgCO2=("emission_kgCO2", "sum"),
    emission_tCO2=("emission_tCO2", "sum"),
    cost_flat=("cost_flat", "sum"),   #EUR
    cost_tou=("cost_tou", "sum"),
    cost_rtp=("cost_rtp", "sum"),
    **{
        f"carbon_value_{cp}": (f"carbon_value_{cp}", "sum")
        for cp in CARBON_PRICES
    }
)

print("Number of daily records:", len(daily))
print("Date range:", daily["date"].min(), "to", daily["date"].max())
print(daily.head())






