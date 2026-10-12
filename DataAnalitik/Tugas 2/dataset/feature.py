import pandas as pd

fire = pd.read_csv("fire.csv")
online = pd.read_csv("online.csv")

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("\n--- NASA FIRMS ---")
print("Rows   :", fire.shape[0])
print("Columns:", fire.shape[1])
print("\nData types:")
print(fire.dtypes)

print("\n--- ONLINE SHOPPERS ---")
print("Rows   :", online.shape[0])
print("Columns:", online.shape[1])
print("\nData types:")
print(online.dtypes)

print("\n" + "=" * 70)
print("COMPLETENESS")
print("=" * 70)

print("\n--- NASA FIRMS ---")
print("Missing values per column:")
print(fire.isnull().sum())

fire_missing = fire.isnull().sum().sum()
fire_total = fire.size
fire_completeness = (1 - fire_missing / fire_total) * 100

print("Total missing values:", fire_missing)
print(f"Completeness: {fire_completeness:.2f}%")

print("\n--- ONLINE SHOPPERS ---")
print("Missing values per column:")
print(online.isnull().sum())

online_missing = online.isnull().sum().sum()
online_total = online.size
online_completeness = (1 - online_missing / online_total) * 100

print("Total missing values:", online_missing)
print(f"Completeness: {online_completeness:.2f}%")

print("\n" + "=" * 70)
print("UNIQUENESS")
print("=" * 70)

print("\n--- NASA FIRMS ---")

fire_duplicates = fire.duplicated().sum()
fire_unique_rows = len(fire) - fire_duplicates

print("Exact duplicate rows:", fire_duplicates)
print("Unique rows:", fire_unique_rows)
print(f"Duplicate percentage: {fire_duplicates / len(fire) * 100:.2f}%")

print("\n--- ONLINE SHOPPERS ---")

online_duplicates = online.duplicated().sum()
online_unique_rows = len(online) - online_duplicates

print("Exact duplicate rows:", online_duplicates)
print("Unique rows:", online_unique_rows)
print(f"Duplicate percentage: {online_duplicates / len(online) * 100:.2f}%")

print("\nFirst duplicated rows in Online Shoppers:")
print(online[online.duplicated(keep=False)].head(10))

print("\n" + "=" * 70)
print("CONSISTENCY")
print("=" * 70)

print("\n--- NASA FIRMS ---")

for col in ["satellite", "instrument", "confidence", "daynight", "type", "version"]:
    if col in fire.columns:
        print(f"\n{col}:")
        print(fire[col].value_counts(dropna=False))

print("\nacq_date sample:")
print(fire["acq_date"].head())

print("\n--- ONLINE SHOPPERS ---")

for col in ["Month", "VisitorType", "Weekend", "Revenue"]:
    if col in online.columns:
        print(f"\n{col}:")
        print(online[col].value_counts(dropna=False))

print("\n" + "=" * 70)
print("VALIDITY")
print("=" * 70)

print("\n--- NASA FIRMS ---")

fire_numeric = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "frp"
]

for col in fire_numeric:
    if col in fire.columns:
        print(f"{col:15s} min={fire[col].min():.5f}  max={fire[col].max():.5f}")

print("\nacq_time:")
print("Min:", fire["acq_time"].min())
print("Max:", fire["acq_time"].max())

def valid_hhmm(x):
    try:
        x = int(x)
        hour = x // 100
        minute = x % 100
        return 0 <= hour <= 23 and 0 <= minute <= 59
    except (ValueError, TypeError):
        return False

fire_invalid_time = ~fire["acq_time"].apply(valid_hhmm)

print("Invalid HHMM values:", fire_invalid_time.sum())


print("\n--- ONLINE SHOPPERS ---")

online_numeric = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType"
]

for col in online_numeric:
    if col in online.columns:
        print(f"{col:25s} min={online[col].min():.5f}  max={online[col].max():.5f}")

print("\n" + "=" * 70)
print("NEGATIVE VALUE CHECK")
print("=" * 70)

print("\n--- NASA FIRMS ---")

for col in fire_numeric:
    if col in fire.columns:
        negative_count = (fire[col] < 0).sum()
        print(f"{col:15s}: {negative_count} negative values")

print("\n--- ONLINE SHOPPERS ---")

for col in online_numeric:
    if col in online.columns:
        negative_count = (online[col] < 0).sum()
        print(f"{col:25s}: {negative_count} negative values")

print("\n" + "=" * 70)
print("TIMELINESS - NASA FIRMS")
print("=" * 70)

fire["acq_date"] = pd.to_datetime(fire["acq_date"], errors="coerce")

print("Start date:", fire["acq_date"].min().date())
print("End date  :", fire["acq_date"].max().date())

unique_dates = fire["acq_date"].dropna().dt.normalize().drop_duplicates()
print("Unique observation dates:", len(unique_dates))

unique_dates = unique_dates.sort_values()

date_gaps = unique_dates.diff().dt.days.dropna()

print("Number of gaps > 1 day:", (date_gaps > 1).sum())
print("Longest gap:", date_gaps.max(), "days")

full_dates = pd.date_range(
    start=fire["acq_date"].min(),
    end=fire["acq_date"].max(),
    freq="D"
)

observed_dates = set(unique_dates)
missing_calendar_days = sum(
    date not in observed_dates
    for date in full_dates
)

print("Calendar days without observation:", missing_calendar_days)

daily_counts = fire.groupby(fire["acq_date"].dt.date).size()

print("\nObservations per day:")
print(daily_counts.describe())

print("\n" + "=" * 70)
print("TIMELINESS - ONLINE SHOPPERS")
print("=" * 70)

print("Month categories:")
print(sorted(online["Month"].dropna().unique()))

print("\nNumber of Month categories:",
    online["Month"].nunique())

print("\nWeekend values:")
print(online["Weekend"].value_counts())

print("\nNo individual timestamp/date column is available.")
print("Missing months:")
all_months = {
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
}

present_months = set(online["Month"].dropna().unique())
missing_months = sorted(all_months - present_months)

print(missing_months)

print("\n" + "=" * 70)
print("UNIQUE VALUE COUNTS")
print("=" * 70)

print("\n--- NASA FIRMS ---")
print(fire.nunique())

print("\n--- ONLINE SHOPPERS ---")
print(online.nunique())

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nNASA FIRMS")
print(f"- Rows               : {len(fire):,}")
print(f"- Columns            : {fire.shape[1]}")
print(f"- Missing values     : {fire_missing}")
print(f"- Exact duplicates   : {fire_duplicates}")
print(f"- Start date         : {fire['acq_date'].min().date()}")
print(f"- End date           : {fire['acq_date'].max().date()}")
print(f"- Unique dates       : {len(unique_dates):,}")
print(f"- Missing calendar days: {missing_calendar_days}")
print(f"- Longest date gap   : {date_gaps.max()} days")

print("\nONLINE SHOPPERS")
print(f"- Rows               : {len(online):,}")
print(f"- Columns            : {online.shape[1]}")
print(f"- Missing values     : {online_missing}")
print(f"- Exact duplicates   : {online_duplicates}")
print(f"- Duplicate %        : {online_duplicates / len(online) * 100:.2f}%")
print(f"- Month categories   : {online['Month'].nunique()}")
print(f"- Missing months     : {missing_months}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)