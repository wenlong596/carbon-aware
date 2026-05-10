
# Carbon-Aware Demand Response in Buildings

##  Project Description

This project focuses on building-level analysis of carbon-aware demand response.

The current version of the repository includes a data processing and scenario construction script, which prepares electricity, tariff, and carbon signals for further analysis.

##  Data

The script requires an input CSV file containing time-series building energy data.

Required columns:

* `timestamp`
* `energy_kWh`
* `wholesale_price_EUR_per_kWh`
* `carbon_intensity_kgCO2_per_kWh`

##  What the Script Does

The script performs the following steps:

### 1. Data preprocessing

* Convert timestamps to timezone-aware format
* Clean and validate key columns
* Generate date and time features

### 2. Tariff construction

* Flat tariff (daily average price)
* Time-of-Use (TOU, Portugal 3-period structure)
* Real-Time Pricing (RTP proxy using wholesale prices)

### 3. Carbon signal construction

* Convert carbon intensity (kgCO₂/kWh → tCO₂/kWh)
* Apply shadow carbon prices (50 / 100 / 200 €/tCO₂)
* Compute carbon cost signals

### 4. Scenario output

* Export processed dataset to:
  `data/processed/*_policy_scenarios.csv`

### 5. Visualization

* Plot daily tariff signals (Flat / TOU / RTP)

### 6. Economic & emission analysis

* Compute:
  * electricity costs (Flat / TOU / RTP)
  * emissions (tCO₂)
  * carbon value under different carbon prices
* Generate daily summary statistics

### 7. Dispatch rule design

* Define three dispatch rules:
  * Rule A: price-first rule
  * Rule B: carbon-first rule
  * Rule C: hybrid rule combining normalized carbon intensity and price
* Support multiple hybrid weights (alpha values)

### 8. Source and target hour identification

* Identify complete 24-hour days
* Rank hours according to the selected dispatch rule
* Mark:
  * avoid / source hours (highest-score hours)
  *  target hours (lowest-score hours)
* Repeat the identification under different tariff scenarios (Flat / TOU / RTP)

### 9. Additional visualization
* Plot source and target hour identification for different dispatch rules
* Compare daily carbon and price signals used for dispatch


