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
# Step 1A - TOU (Portugal Continental 3-period, ERSE weekly cycle)
# =========================
# Based on ERSE Continental / Weekly / 24-hour timetable.
#
# ERSE original periods:
#   Super off-peak, Off-peak, Mid-peak, Peak
#
# Three-period mapping in this study:
#   offpeak  = Super off-peak + Off-peak
#   shoulder = Mid-peak
#   peak     = Peak
#
# Timetable:
#   Summer = June to October
#   Winter = November to May


def is_erse_summer(month: int) -> bool:
    """
    ERSE timetable:
    Summer: June to October
    Winter: November to May
    """
    return month in [6, 7, 8, 9, 10]


def tou_portugal_3period_erse(ts) -> str:
    """
    ERSE-based 3-period TOU mapping for Portugal Continental,
    weekly cycle, 24-hour format.

    Returns:
        "offpeak", "shoulder", or "peak"
    """
    ts = pd.Timestamp(ts)

    dow = ts.dayofweek  # Monday=0, ..., Sunday=6
    m = ts.hour * 60 + ts.minute
    summer = is_erse_summer(ts.month)

    # Sunday:
    # ERSE shows only super off-peak/off-peak.
    # Since both are merged in this study, Sunday is fully offpeak.
    if dow == 6:
        return "offpeak"

    # Saturday
    if dow == 5:
        if summer:
            # Summer Saturday:
            # 00:00-09:00 offpeak
            # 09:00-14:00 shoulder
            # 14:00-20:00 offpeak
            # 20:00-22:00 shoulder
            # 22:00-24:00 offpeak
            if (9 * 60 <= m < 14 * 60) or (20 * 60 <= m < 22 * 60):
                return "shoulder"
            return "offpeak"

        else:
            # Winter Saturday:
            # 00:00-09:30 offpeak
            # 09:30-13:00 shoulder
            # 13:00-18:30 offpeak
            # 18:30-22:00 shoulder
            # 22:00-24:00 offpeak
            if (9 * 60 + 30 <= m < 13 * 60) or (18 * 60 + 30 <= m < 22 * 60):
                return "shoulder"
            return "offpeak"

    # Monday to Friday
    if summer:
        # Summer weekdays:
        # 00:00-07:00 offpeak
        # 07:00-09:15 shoulder
        # 09:15-12:15 peak
        # 12:15-24:00 shoulder
        if m < 7 * 60:
            return "offpeak"
        if 9 * 60 + 15 <= m < 12 * 60 + 15:
            return "peak"
        return "shoulder"

    else:
        # Winter weekdays:
        # 00:00-07:00 offpeak
        # 07:00-09:30 shoulder
        # 09:30-12:00 peak
        # 12:00-18:30 shoulder
        # 18:30-21:00 peak
        # 21:00-24:00 shoulder
        if m < 7 * 60:
            return "offpeak"
        if (9 * 60 + 30 <= m < 12 * 60) or (18 * 60 + 30 <= m < 21 * 60):
            return "peak"
        return "shoulder"


df["tou_period"] = df[TIME_COL].apply(tou_portugal_3period_erse)


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
#print(daily.head())


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

    Rule A: price-first rule
    Rule B: carbon-first rule
    Rule C: hybrid of normalized carbon + normalized tariff-specific price

    For Rule C:
    alpha = 0   -> price-only
    alpha = 1   -> carbon-only
    """
    if rule == "A":
        if price_col is None:
            raise ValueError("price_col must be provided for Rule A.")
        return day_df[price_col].astype(float)

    elif rule == "B":
        return day_df[CI_COL].astype(float)

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
    - Rule A is price-first rule, so it uses tariff-specific price columns.
    - Rule B is carbon-first rule, so its score does not depend on tariff.
    - Rule C uses both tariff-specific price columns and carbon intensity.
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

            # Rule A: price-first rule 
            sA = compute_score(day_df, rule="A", price_col=price_col)
            avoidA = label_top_hours(day_df, sA, top_pct=top_pct)
            targetA = label_bottom_hours(day_df, sA, top_pct=top_pct)

            out.append(pd.DataFrame({
                time_col: day_df[time_col].tolist(),
                "date": day_df["date"].values,
                "hour": day_df["hour"].values,
                "tariff": tariff_name,
                "rule": "A",
                "alpha": np.nan,
                "score": sA.values,
                "avoid": avoidA.values,
                "target": targetA.values
            }))

            # Rule B: carbon-first rule 
            sB = compute_score(day_df, rule="B")
            avoidB = label_top_hours(day_df, sB, top_pct=top_pct)
            targetB = label_bottom_hours(day_df, sB, top_pct=top_pct)

            out.append(pd.DataFrame({
                time_col: day_df[time_col].tolist(),
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
                    time_col: day_df[time_col].tolist(),
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

    #r = step3_res.copy()
    #r[time_col] = pd.to_datetime(r[time_col], errors="coerce")
    #r["date"] = pd.to_datetime(r[time_col]).dt.date

    r = step3_res.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.date

    

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
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]}
    )

    def draw_rule(ax, sub, label, ylabel):
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

        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)


    draw_rule(axes[0], subA, "Rule A", "Rule A Price\n(EUR/kWh)")
    draw_rule(axes[1], subB, "Rule B", "Rule B CI\n(kgCO₂/kWh)")
    draw_rule(axes[2], subC, "Rule C", "Rule C Hybrid\n(normalized)")
    

    axes[0].set_title(
        f"Source and target hour identification under {tariff.upper()} tariff ({plot_day})"
    )

    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(25))
    axes[-1].set_xlabel("Hour")

    plt.tight_layout()
    plt.show()


# =========================
# Plot Step 3 for the SAME selected day
# =========================
def plot_step3_for_selected_day(df, step3_res, plot_day):

    plot_step3_day_0_24h(
        step3_res=step3_res,
        original_df=df,
        plot_day=plot_day,
        tariff="tou",
        alpha=0.5
    )

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



# =========================
# Step 4 - Constrained Load-Shifting Simulation
# =========================

FLEX_SHARES = [0.10, 0.15, 0.20]

# Shifted load cannot exceed 110% of the original daily peak
PEAK_CAP_FACTOR = 1.10

# Ramp constraint:
# shifted ramp cannot exceed RAMP_LIMIT_FACTOR times the maximum baseline daily ramp
RAMP_LIMIT_FACTOR = 1.50

# Maximum consecutive controlled hours
MAX_CONSECUTIVE_CONTROL_HOURS = 3


def limit_consecutive_hours(mask, scores, max_hours, mode="source"):
    """
    Limit consecutive True values in a boolean mask.

    mode="source": keep highest-score hours within a consecutive block.
    mode="target": keep lowest-score hours within a consecutive block.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    scores = np.asarray(scores, dtype=float)

    if max_hours is None:
        return mask

    n = len(mask)
    new_mask = mask.copy()
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i
        while i < n and mask[i]:
            i += 1
        end = i

        block_indices = np.arange(start, end)

        if len(block_indices) > max_hours:
            if mode == "source":
                keep = block_indices[np.argsort(scores[block_indices])[-max_hours:]]
            elif mode == "target":
                keep = block_indices[np.argsort(scores[block_indices])[:max_hours]]
            else:
                raise ValueError("mode must be 'source' or 'target'")

            remove = np.setdiff1d(block_indices, keep)
            new_mask[remove] = False

    return new_mask


def check_ramp_constraint(load, ramp_limit_kWh):
    """
    Check whether shifted load satisfies the ramping constraint.
    """
    if ramp_limit_kWh is None:
        return True

    ramps = np.abs(np.diff(load))
    return np.all(ramps <= ramp_limit_kWh + 1e-9)


def max_feasible_pair_shift(
    L0,
    x,
    y,
    src_idx,
    tgt_idx,
    requested_delta,
    daily_peak_cap,
    ramp_limit_kWh,
    inflexible
):
    """
    Find the maximum feasible amount that can be shifted from one source hour
    to one target hour while satisfying:
    - non-negative flexible reduction
    - target peak cap
    - ramp constraint
    - inflexible-load lower bound
    """

    if requested_delta <= 0:
        return 0.0

    # Upper bound due to target peak cap
    target_capacity = daily_peak_cap - (L0[tgt_idx] + y[tgt_idx])
    target_capacity = max(0.0, target_capacity)

    hi = min(requested_delta, target_capacity)

    if hi <= 0:
        return 0.0

    def feasible(delta):
        x_try = x.copy()
        y_try = y.copy()

        x_try[src_idx] += delta
        y_try[tgt_idx] += delta

        L_try = L0 - x_try + y_try

        if np.any(L_try < inflexible - 1e-9):
            return False

        if np.any(L_try > daily_peak_cap + 1e-9):
            return False

        if not check_ramp_constraint(L_try, ramp_limit_kWh):
            return False

        return True

    # If full amount is feasible, use it
    if feasible(hi):
        return hi

    # Otherwise use binary search
    lo = 0.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid

    return lo


def run_step4_load_shifting(
    df: pd.DataFrame,
    step3_res: pd.DataFrame,
    flex_shares: list = FLEX_SHARES,
    peak_cap_factor: float = PEAK_CAP_FACTOR,
    ramp_limit_factor: float = RAMP_LIMIT_FACTOR,
    max_consecutive_control_hours: int = MAX_CONSECUTIVE_CONTROL_HOURS,
    time_col: str = TIME_COL,
    load_col: str = E_COL,
    ci_col: str = CI_COL,
    price_cols: dict = PRICE_COLS
) -> pd.DataFrame:

    base = df.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base["date"] = base[time_col].dt.date
    base["hour"] = base[time_col].dt.hour

    base[load_col] = pd.to_numeric(base[load_col], errors="coerce")
    base[ci_col] = pd.to_numeric(base[ci_col], errors="coerce")

    required_base_cols = [time_col, "date", "hour", load_col, ci_col] + list(price_cols.values())
    base = base.dropna(subset=required_base_cols).copy()

    r = step3_res.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.date

    out = []
    group_cols = ["date", "tariff", "rule", "alpha"]

    for (date, tariff, rule, alpha), rule_df in r.groupby(group_cols, dropna=False):

        day_base = base[base["date"] == date].sort_values("hour").copy()
        rule_day = rule_df.sort_values("hour").copy()

        if len(day_base) != 24 or day_base["hour"].nunique() != 24:
            continue

        if len(rule_day) != 24 or rule_day["hour"].nunique() != 24:
            continue

        price_col = price_cols[tariff]

        day = day_base.merge(
            rule_day[[time_col, "tariff", "rule", "alpha", "score", "avoid", "target"]],
            on=time_col,
            how="inner"
        )

        if len(day) != 24:
            continue

        day = day.sort_values("hour").reset_index(drop=True)

        L0 = day[load_col].astype(float).to_numpy()
        score = day["score"].astype(float).to_numpy()

        avoid_mask_raw = day["avoid"].astype(bool).to_numpy()
        target_mask_raw = day["target"].astype(bool).to_numpy()

        # Apply maximum consecutive control duration
        avoid_mask = limit_consecutive_hours(
            avoid_mask_raw,
            score,
            max_consecutive_control_hours,
            mode="source"
        )

        target_mask = limit_consecutive_hours(
            target_mask_raw,
            score,
            max_consecutive_control_hours,
            mode="target"
        )

        daily_peak_cap = peak_cap_factor * np.nanmax(L0)

        baseline_ramps = np.abs(np.diff(L0))
        max_baseline_ramp = np.nanmax(baseline_ramps)

        if np.isclose(max_baseline_ramp, 0.0):
            ramp_limit_kWh = None
        else:
            ramp_limit_kWh = ramp_limit_factor * max_baseline_ramp

        for lam in flex_shares:

            flexible = lam * L0
            inflexible = (1 - lam) * L0

            x = np.zeros(len(day))
            y = np.zeros(len(day))

            # Source hours: highest score first
            source_order = (
                day.loc[avoid_mask]
                   .sort_values("score", ascending=False)
                   .index
                   .to_list()
            )

            # Target hours: lowest score first
            target_order = (
                day.loc[target_mask]
                   .sort_values("score", ascending=True)
                   .index
                   .to_list()
            )

            if len(source_order) == 0 or len(target_order) == 0:
                continue

            total_shift_potential = flexible[avoid_mask].sum()

            for src_idx in source_order:

                source_remaining = flexible[src_idx] - x[src_idx]

                if source_remaining <= 1e-9:
                    continue

                for tgt_idx in target_order:

                    if source_remaining <= 1e-9:
                        break

                    delta = max_feasible_pair_shift(
                        L0=L0,
                        x=x,
                        y=y,
                        src_idx=src_idx,
                        tgt_idx=tgt_idx,
                        requested_delta=source_remaining,
                        daily_peak_cap=daily_peak_cap,
                        ramp_limit_kWh=ramp_limit_kWh,
                        inflexible=inflexible
                    )

                    if delta <= 1e-9:
                        continue

                    x[src_idx] += delta
                    y[tgt_idx] += delta
                    source_remaining -= delta

            L_shift = L0 - x + y

            energy_diff = L_shift.sum() - L0.sum()

            result = day.copy()

            result["flex_share"] = lam
            result["L0_kWh"] = L0
            result["flexible_kWh"] = flexible
            result["inflexible_kWh"] = inflexible

            result["avoid_raw"] = avoid_mask_raw
            result["target_raw"] = target_mask_raw
            result["avoid_limited"] = avoid_mask
            result["target_limited"] = target_mask

            result["x_reduced_kWh"] = x
            result["y_added_kWh"] = y
            result["L_shifted_kWh"] = L_shift

            result["daily_peak_cap_kWh"] = daily_peak_cap
            result["ramp_limit_kWh"] = ramp_limit_kWh
            result["max_consecutive_control_hours"] = max_consecutive_control_hours

            result["shift_potential_kWh"] = total_shift_potential
            result["shift_out_kWh"] = x.sum()
            result["shift_in_kWh"] = y.sum()
            result["unallocated_shift_kWh"] = total_shift_potential - x.sum()
            result["energy_difference_kWh"] = energy_diff

            result["ramp_violation"] = not check_ramp_constraint(L_shift, ramp_limit_kWh)
            result["peak_violation"] = np.any(L_shift > daily_peak_cap + 1e-9)

            result["emission_base_kgCO2"] = result["L0_kWh"] * result[ci_col]
            result["emission_shifted_kgCO2"] = result["L_shifted_kWh"] * result[ci_col]

            result["price_used_EUR_per_kWh"] = result[price_col]
            result["cost_base_EUR"] = result["L0_kWh"] * result["price_used_EUR_per_kWh"]
            result["cost_shifted_EUR"] = result["L_shifted_kWh"] * result["price_used_EUR_per_kWh"]

            out.append(result)

    if len(out) == 0:
        raise ValueError("No shifted load results were generated. Check Step 3 output and complete days.")

    return pd.concat(out, ignore_index=True)


# =========================
# Run Step 4
# =========================


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 200)

shifted_res = run_step4_load_shifting(df, step3_res)

#print("Step 4 shifted results:")
#print(shifted_res.head())

print("\nNumber of rows:", len(shifted_res))
print("Tariffs:", shifted_res["tariff"].unique())
print("Rules:", shifted_res["rule"].unique())
print("Flex shares:", shifted_res["flex_share"].unique())

print("\nRows by tariff:")
print(shifted_res["tariff"].value_counts())

#print("\nRows by tariff and rule:")
#print(shifted_res.groupby(["tariff", "rule"]).size())


# =========================
# Step 4A - Daily Summary
# =========================

daily_shift_summary = shifted_res.groupby(
    ["date", "tariff", "rule", "alpha", "flex_share"],
    dropna=False,
    as_index=False
).agg(
    L0_kWh=("L0_kWh", "sum"),
    L_shifted_kWh=("L_shifted_kWh", "sum"),
    shift_potential_kWh=("shift_potential_kWh", "max"),
    shift_out_kWh=("x_reduced_kWh", "sum"),
    shift_in_kWh=("y_added_kWh", "sum"),
    unallocated_shift_kWh=("unallocated_shift_kWh", "max"),
    emission_base_kgCO2=("emission_base_kgCO2", "sum"),
    emission_shifted_kgCO2=("emission_shifted_kgCO2", "sum"),
    cost_base_EUR=("cost_base_EUR", "sum"),
    cost_shifted_EUR=("cost_shifted_EUR", "sum"),
    max_L0_kWh=("L0_kWh", "max"),
    max_L_shifted_kWh=("L_shifted_kWh", "max"),
    daily_peak_cap_kWh=("daily_peak_cap_kWh", "max"),
    ramp_limit_kWh=("ramp_limit_kWh", "max"),
    ramp_violation=("ramp_violation", "max"),
    peak_violation=("peak_violation", "max"),
    max_energy_difference_kWh=("energy_difference_kWh", "max")
)

daily_shift_summary["emission_reduction_kgCO2"] = (
    daily_shift_summary["emission_base_kgCO2"]
    - daily_shift_summary["emission_shifted_kgCO2"]
)

daily_shift_summary["cost_change_EUR"] = (
    daily_shift_summary["cost_shifted_EUR"]
    - daily_shift_summary["cost_base_EUR"]
)

daily_shift_summary["energy_conservation_error_kWh"] = (
    daily_shift_summary["L_shifted_kWh"]
    - daily_shift_summary["L0_kWh"]
)

#print("\nDaily shift summary:")
#print(daily_shift_summary.head())

print("\nMax absolute daily energy difference:")
print(daily_shift_summary["energy_conservation_error_kWh"].abs().max())

print("\nRamp violations:")
print(daily_shift_summary["ramp_violation"].sum())

print("\nPeak violations:")
print(daily_shift_summary["peak_violation"].sum())


# =========================
# Step 4B - Summary by Case
# =========================

summary_by_case = daily_shift_summary.groupby(
    ["tariff", "rule", "alpha", "flex_share"],
    dropna=False,
    as_index=False
).agg(
    mean_emission_reduction_kgCO2=("emission_reduction_kgCO2", "mean"),
    total_emission_reduction_kgCO2=("emission_reduction_kgCO2", "sum"),
    mean_cost_change_EUR=("cost_change_EUR", "mean"),
    total_cost_change_EUR=("cost_change_EUR", "sum"),
    mean_shift_potential_kWh=("shift_potential_kWh", "mean"),
    mean_shift_out_kWh=("shift_out_kWh", "mean"),
    mean_shift_in_kWh=("shift_in_kWh", "mean"),
    mean_unallocated_shift_kWh=("unallocated_shift_kWh", "mean"),
    mean_max_L0_kWh=("max_L0_kWh", "mean"),
    mean_max_L_shifted_kWh=("max_L_shifted_kWh", "mean")
)

from pathlib import Path

# =========================
# Save summary tables
# =========================

RESULT_DIR = Path(r"C:\Users\LIUWENLONG\Desktop\carbon-aware-dr-wenlong\result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

summary_by_case.to_csv(
    RESULT_DIR / "summary_by_case.csv",
    index=False
)

daily_shift_summary.to_csv(
    RESULT_DIR / "daily_shift_summary.csv",
    index=False
)

print("\nSaved:")
print(RESULT_DIR / "summary_by_case.csv")
print(RESULT_DIR / "daily_shift_summary.csv")

# =========================
# Baseline vs shifted load
# =========================

def plot_baseline_vs_shifted(
    shifted_res,
    plot_day,
    tariff="rtp",
    rule="C",
    alpha=0.5,
    flex_share=0.15,
    time_col="timestamp"
):
    d = shifted_res.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d["date"] = pd.to_datetime(d["date"]).dt.date

    plot_day = pd.to_datetime(plot_day).date()

    if rule == "C":
        sub = d[
            (d["date"] == plot_day) &
            (d["tariff"] == tariff) &
            (d["rule"] == rule) &
            (np.isclose(d["alpha"], alpha)) &
            (np.isclose(d["flex_share"], flex_share))
        ].sort_values("hour")
    else:
        sub = d[
            (d["date"] == plot_day) &
            (d["tariff"] == tariff) &
            (d["rule"] == rule) &
            (np.isclose(d["flex_share"], flex_share))
        ].sort_values("hour")

    if len(sub) != 24:
        raise ValueError("No complete 24-hour result found for the selected case.")

    x = sub["hour"].tolist() + [24]
    y_base = sub["L0_kWh"].tolist() + [sub["L0_kWh"].iloc[-1]]
    y_shift = sub["L_shifted_kWh"].tolist() + [sub["L_shifted_kWh"].iloc[-1]]

    plt.figure(figsize=(11, 5))
    plt.plot(x, y_base, drawstyle="steps-post", label="Baseline load", linewidth=2)
    plt.plot(x, y_shift, drawstyle="steps-post", label="Shifted load", linewidth=2)

    for h in sub.loc[sub["avoid_limited"], "hour"]:
        plt.axvspan(h, h + 1, alpha=0.12)

    for h in sub.loc[sub["target_limited"], "hour"]:
        plt.axvspan(h, h + 1, alpha=0.08)

    title_alpha = f", alpha={alpha}" if rule == "C" else ""

    plt.title(
        f"Baseline vs shifted load under {tariff.upper()} tariff, "
        f"Rule {rule}{title_alpha}, flex={flex_share} ({plot_day})"
    )
    plt.xlabel("Hour of day")
    plt.ylabel("Electricity load (kWh)")
    plt.xticks(range(25))
    plt.xlim(0, 24)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_baseline_vs_shifted(
    shifted_res=shifted_res,
    plot_day=plot_day,
    tariff="rtp",
    rule="C",
    alpha=0.5,
    flex_share=0.15
)

# =========================
# Emission reduction vs alpha
# =========================

def plot_emission_reduction_vs_alpha(summary_by_case, tariff="rtp"):
    d = summary_by_case.copy()

    sub = d[
        (d["tariff"] == tariff) &
        (d["rule"] == "C")
    ].sort_values(["flex_share", "alpha"])

    plt.figure(figsize=(8, 5))

    for flex in sorted(sub["flex_share"].unique()):
        s = sub[sub["flex_share"] == flex]
        plt.plot(
            s["alpha"],
            s["total_emission_reduction_kgCO2"],
            marker="o",
            label=f"flex={flex}"
        )

    plt.title(f"Emission reduction under {tariff.upper()} tariff")
    plt.xlabel("Carbon weight alpha")
    plt.ylabel("Total emission reduction (kgCO₂)")
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_emission_reduction_vs_alpha(summary_by_case, tariff="tou")
plot_emission_reduction_vs_alpha(summary_by_case, tariff="rtp")

# =========================
# Cost change vs alpha
# =========================

def plot_cost_change_vs_alpha(summary_by_case, tariff="rtp"):
    d = summary_by_case.copy()

    sub = d[
        (d["tariff"] == tariff) &
        (d["rule"] == "C")
    ].sort_values(["flex_share", "alpha"])

    plt.figure(figsize=(8, 5))

    for flex in sorted(sub["flex_share"].unique()):
        s = sub[sub["flex_share"] == flex]
        plt.plot(
            s["alpha"],
            s["total_cost_change_EUR"],
            marker="o",
            label=f"flex={flex}"
        )

    plt.axhline(0, linewidth=1)
    plt.title(f"Cost change under {tariff.upper()} tariff")
    plt.xlabel("Carbon weight alpha")
    plt.ylabel("Total cost change (EUR)")
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_cost_change_vs_alpha(summary_by_case, tariff="tou")
plot_cost_change_vs_alpha(summary_by_case, tariff="rtp")

# =========================
# Cost–emission trade-off plot
# =========================

def plot_cost_emission_tradeoff(summary_by_case, flex_share=0.20):
    d = summary_by_case.copy()

    sub = d[
        (d["rule"] == "C") &
        (np.isclose(d["flex_share"], flex_share)) &
        (d["tariff"].isin(["flat", "tou", "rtp"]))
    ].sort_values(["tariff", "alpha"])

    plt.figure(figsize=(8, 6))

    for tariff in ["flat", "tou", "rtp"]:
        s = sub[sub["tariff"] == tariff]
        plt.plot(
            s["total_cost_change_EUR"],
            s["total_emission_reduction_kgCO2"],
            marker="o",
            label=tariff.upper()
        )

        for _, row in s.iterrows():
            plt.annotate(
                f"{row['alpha']:.2f}",
                (row["total_cost_change_EUR"], row["total_emission_reduction_kgCO2"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    plt.axvline(0, linewidth=1)
    plt.title(f"Cost-emission trade-off, flex={flex_share}")
    plt.xlabel("Total cost change (EUR)")
    plt.ylabel("Total emission reduction (kgCO₂)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tariff")
    plt.tight_layout()
    plt.show()


plot_cost_emission_tradeoff(summary_by_case, flex_share=0.20)