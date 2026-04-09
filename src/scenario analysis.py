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

# Basic column validation
need_cols = [TIME_COL, E_COL, P_COL, CI_KG_COL]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Your columns are: {list(df.columns)}")

df["date"] = df[TIME_COL].dt.date
df["dow"] = df[TIME_COL].dt.dayofweek  # Mon=0 ... Sun=6
df["minute_of_day"] = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute

# Convert to numeric values
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

# Generate "carbon_cost_EUR" under different carbon price scenarios
for cp in CARBON_PRICES:
    df[f"carbon_price_EUR_per_tCO2_{cp}"] = cp
    # # Carbon cost (€) = energy(kWh) * tCO2/kWh * €/tCO2
    df[f"carbon_cost_EUR_{cp}"] = df[E_COL] * df["tCO2_per_kWh"] * cp

# Carbon cost signal per kWh (€/kWh)
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

# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # Project root directory

# Get input file name
base_name = os.path.basename(CSV_PATH)
name_no_ext = os.path.splitext(base_name)[0]

# Generate output file name
out_file = f"{name_no_ext}_policy_scenarios.csv"

# Construct file path
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

# 1) Select one "complete day" (24h) for plotting
daily_counts = df.groupby("date")[TIME_COL].count().sort_values(ascending=False)
valid_days = daily_counts[daily_counts >= 24].index  # 至少2条（1h数据=24条，15min=96条）
if len(valid_days) == 0:
    # Fallback: select the first available day
    plot_day = df["date"].iloc[0]
else:
    #plot_day = valid_days[len(valid_days)//2]  Select the median day
    plot_day = np.random.choice(valid_days) # Randomly select one day
    
day_df = df[df["date"] == plot_day].copy()


# Fig 1: Flat / TOU / RTP signals for the same day
day_df = day_df.sort_values(TIME_COL).copy()

# Extract hour
day_df["hour"] = day_df[TIME_COL].dt.hour

# If multiple values exist within the same hour, take the hourly average
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


# =========================
# Step 3 - Dispatch Rules 
# Output: for each complete 24h day, identify avoid/source hours and target hours
# for Rule A / B / C under each tariff scenario
# =========================


# EDIT column names if needed 
TIME_COL = "timestamp"
E_COL    = "energy_kWh"   
CI_COL   = "carbon_intensity_kgCO2_per_kWh"

# tariff-specific price columns
PRICE_COLS = {
    "flat": "price_flat",
    "tou":  "price_tou",
    "rtp":  "price_rtp"
}

TOP_PCT = 0.15
ALPHAS  = [0.0, 0.25, 0.5, 0.75, 1.0]   # Rule C only


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx - mn, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def compute_score(day_df: pd.DataFrame, rule: str, price_col: str = None, alpha: float = 0.5) -> pd.Series:
    """
    Higher score = worse = avoid/source priority
    Rule A: carbon-first rule
    Rule B: Price-first rule (tariff-specific)
    Rule C: hybrid of normalized carbon + normalized tariff-specific price
    """
    if rule == "A":
        return day_df[CI_COL].astype(float)

    elif rule == "B":
        if price_col is None:
            raise ValueError("price_col must be provided for Rule B.")
        return day_df[price_col].astype(float)

    elif rule == "C":
        if price_col is None:
            raise ValueError("price_col must be provided for Rule C.")
        ci_n = _minmax(day_df[CI_COL])
        p_n  = _minmax(day_df[price_col])
        return alpha * ci_n + (1 - alpha) * p_n

    else:
        raise ValueError("rule must be 'A', 'B', or 'C'")


def label_top_hours(day_df: pd.DataFrame, score: pd.Series, top_pct: float = 0.15) -> pd.Series:
    """
    True = avoid/source hours (highest score)
    """
    n = len(day_df)
    n_sel = max(1, int(np.ceil(n * top_pct)))
    top_idx = score.nlargest(n_sel).index
    return pd.Series(day_df.index.isin(top_idx), index=day_df.index)


def label_bottom_hours(day_df: pd.DataFrame, score: pd.Series, top_pct: float = 0.15) -> pd.Series:
    """
    True = target hours (lowest score)
    """
    n = len(day_df)
    n_sel = max(1, int(np.ceil(n * top_pct)))
    bottom_idx = score.nsmallest(n_sel).index
    return pd.Series(day_df.index.isin(bottom_idx), index=day_df.index)


def run_step3_rules(df: pd.DataFrame,
                    time_col: str = TIME_COL,
                    ci_col: str = CI_COL,
                    price_cols: dict = PRICE_COLS,
                    top_pct: float = TOP_PCT,
                    alphas: list = ALPHAS) -> pd.DataFrame:
    """
    Runs Step 3 for all complete 24h days and all tariff scenarios.

    Returns a long-format dataframe with columns:
      timestamp, date, hour, tariff, rule, alpha, score, avoid, target

    Notes:
    - Rule A is carbon-first rule, so score itself does not depend on tariff.
      But we still repeat it under each tariff for easier downstream merging/comparison.
    - Rule B and Rule C use tariff-specific price columns.
    """

    required_cols = [time_col, ci_col] + list(price_cols.values())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=required_cols).copy()
    d = d.sort_values(time_col)

    d["date"] = d[time_col].dt.date
    d["hour"] = d[time_col].dt.hour

    # complete 24h days: exactly 24 rows and 24 unique hours
    day_stats = (
        d.groupby("date")
         .agg(n_rows=(time_col, "count"),
              n_hours=("hour", "nunique"))
         .reset_index()
    )
    complete_days = day_stats[
        (day_stats["n_rows"] == 24) & (day_stats["n_hours"] == 24)
    ]["date"].tolist()

    if len(complete_days) == 0:
        raise ValueError("No complete 24h days found. Check missing/duplicate hours or aggregation.")

    out = []

    for day in complete_days:
        day_df = d[d["date"] == day].sort_values(time_col).reset_index(drop=True)

        for tariff_name, price_col in price_cols.items():

            # Rule A: carbon-first rule 
            sA = compute_score(day_df, rule="A")
            avoidA = label_top_hours(day_df, sA, top_pct=top_pct)
            targetA = label_bottom_hours(day_df, sA, top_pct=top_pct)

            out.append(pd.DataFrame({
                time_col: day_df[time_col].values,
                "date": day_df["date"].values,
                "hour": day_df["hour"].values,
                "tariff": tariff_name,
                "rule": "A",
                "alpha": np.nan,
                "score": sA.values,
                "avoid": avoidA.values,
                "target": targetA.values
            }))

            # Rule B: Price-first rule 
            sB = compute_score(day_df, rule="B", price_col=price_col)
            avoidB = label_top_hours(day_df, sB, top_pct=top_pct)
            targetB = label_bottom_hours(day_df, sB, top_pct=top_pct)

            out.append(pd.DataFrame({
                time_col: day_df[time_col].values,
                "date": day_df["date"].values,
                "hour": day_df["hour"].values,
                "tariff": tariff_name,
                "rule": "B",
                "alpha": np.nan,
                "score": sB.values,
                "avoid": avoidB.values,
                "target": targetB.values
            }))

            # Rule C: hybrid rule 
            for a in alphas:
                sC = compute_score(day_df, rule="C", price_col=price_col, alpha=a)
                avoidC = label_top_hours(day_df, sC, top_pct=top_pct)
                targetC = label_bottom_hours(day_df, sC, top_pct=top_pct)

                out.append(pd.DataFrame({
                    time_col: day_df[time_col].values,
                    "date": day_df["date"].values,
                    "hour": day_df["hour"].values,
                    "tariff": tariff_name,
                    "rule": "C",
                    "alpha": a,
                    "score": sC.values,
                    "avoid": avoidC.values,
                    "target": targetC.values
                }))

    res = pd.concat(out, ignore_index=True)
    return res



# =========================
# Step 3 visualization（0–24h）
# =========================
def plot_step3_day_0_24h(step3_res,
                         original_df,
                         plot_day,
                         tariff="tou",
                         alpha=0.5,
                         time_col="timestamp",
                         ci_col="carbon_intensity_kgCO2_per_kWh",
                         price_cols=None):

    if price_cols is None:
        price_cols = {
            "flat": "price_flat",
            "tou": "price_tou",
            "rtp": "price_rtp"
        }

    price_col = price_cols[tariff]

    d0 = original_df.copy()
    d0[time_col] = pd.to_datetime(d0[time_col], errors="coerce")
    d0 = d0.dropna(subset=[time_col, ci_col, price_col]).copy()

    d0["date"] = d0[time_col].dt.date
    d0["hour"] = d0[time_col].dt.hour

    plot_day = pd.to_datetime(plot_day).date()
    day_df = d0[d0["date"] == plot_day].sort_values(time_col).copy()

    if len(day_df) != 24 or day_df["hour"].nunique() != 24:
        raise ValueError(f"{plot_day} is not a complete 24h day.")

    def extend_to_24(x, y):
        x_ext = list(x) + [24]
        y_ext = list(y) + [y.iloc[-1]]
        return x_ext, y_ext

    r = step3_res.copy()
    r[time_col] = pd.to_datetime(r[time_col], errors="coerce")
    r["date"] = pd.to_datetime(r[time_col]).dt.date

    subA = r[
        (r["date"] == plot_day) &
        (r["tariff"] == tariff) &
        (r["rule"] == "A")
    ].sort_values("hour")

    subB = r[
        (r["date"] == plot_day) &
        (r["tariff"] == tariff) &
        (r["rule"] == "B")
    ].sort_values("hour")

    subC = r[
        (r["date"] == plot_day) &
        (r["tariff"] == tariff) &
        (r["rule"] == "C") &
        (np.isclose(r["alpha"], alpha))
    ].sort_values("hour")

    fig, axes = plt.subplots(
        5, 1, figsize=(12, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1, 1]}
    )

    x_ci, y_ci = extend_to_24(day_df["hour"], day_df[ci_col])
    axes[0].plot(x_ci, y_ci, drawstyle="steps-post")
    axes[0].set_ylabel("CI")
    axes[0].set_title(f"Source and target hour identification under {tariff.upper()} tariff ({plot_day})")

    x_p, y_p = extend_to_24(day_df["hour"], day_df[price_col])
    axes[1].plot(x_p, y_p, drawstyle="steps-post")
    axes[1].set_ylabel("Price")

    def draw_rule(ax, sub, label):
        x_s, y_s = extend_to_24(sub["hour"], sub["score"])
        ax.plot(x_s, y_s, drawstyle="steps-post")

        avoid = sub[sub["avoid"]]
        target = sub[sub["target"]]

        ax.scatter(avoid["hour"] + 0.5, avoid["score"], label="avoid", s=50)
        ax.scatter(target["hour"] + 0.5, target["score"], label="target", s=50)

        for h in avoid["hour"]:
            ax.axvspan(h, h + 1, alpha=0.15)
        for h in target["hour"]:
            ax.axvspan(h, h + 1, alpha=0.10)

        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    draw_rule(axes[2], subA, "Rule A")
    draw_rule(axes[3], subB, "Rule B")
    draw_rule(axes[4], subC, "Rule C")

    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(25))
    axes[-1].set_xlabel("Hour")

    plt.tight_layout()
    plt.show()


# =========================
# Plot Step 3 for the SAME selected day
# =========================
def plot_step3_for_selected_day(df, step3_res, plot_day):

    #print("\n--- TOU ---")
    plot_step3_day_0_24h(
        step3_res=step3_res,
        original_df=df,
        plot_day=plot_day,
        tariff="tou",
        alpha=0.5
    )

    #print("\n--- RTP ---")
    plot_step3_day_0_24h(
        step3_res=step3_res,
        original_df=df,
        plot_day=plot_day,
        tariff="rtp",
        alpha=0.5
    )

# =========================
# Run Step 3 + Plot
# =========================
step3_res = run_step3_rules(df)
plot_step3_for_selected_day(df, step3_res, plot_day)



