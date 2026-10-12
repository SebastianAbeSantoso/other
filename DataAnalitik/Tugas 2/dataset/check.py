import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

fire = pd.read_csv("fire.csv")

online = pd.read_csv("online.csv")
for name, df in [("NASA FIRMS", fire), ("Online Shoppers", online)]:
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("Data types:\n", df.dtypes)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Descriptive statistics:\n", df.describe().T.round(2))

fire["acq_date"] = pd.to_datetime(fire["acq_date"], errors="coerce")
fire["acq_month"] = fire["acq_date"].dt.month
fire["acq_day_of_week"] = fire["acq_date"].dt.day_name()

online_cleaned = online.drop_duplicates().copy()
online_cleaned["TotalPages"] = online_cleaned[
    ["Administrative", "Informational", "ProductRelated"]
].sum(axis=1)

online_cleaned["TotalDuration"] = online_cleaned[
    [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]
].sum(axis=1)

print("\n--- NASA ENGINEERED FEATURES ---")
print(fire[["acq_date", "acq_month", "acq_day_of_week"]].head())

print("\n--- ONLINE SHOPPERS ENGINEERED FEATURES ---")
print(online_cleaned[["TotalPages", "TotalDuration", "Revenue"]].head())
print("Rows after removing duplicates:", len(online_cleaned))
fire_skew = fire.select_dtypes(include="number").skew().sort_values()
online_skew = online_cleaned.select_dtypes(include="number").skew().sort_values()

print("\n--- NASA FIRMS: SKEWNESS ---")
print(fire_skew.round(3))
print("\n--- ONLINE SHOPPERS: SKEWNESS ---")
print(online_skew.round(3))

def check_outliers(df, name):
    numeric = df.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    result = pd.DataFrame(
        {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outlier Count": ((numeric < lower) | (numeric > upper)).sum(),
        }
    )

    print(f"\n--- {name}: IQR OUTLIER CHECK ---")
    print(result.round(3).to_string())

check_outliers(fire, "NASA FIRMS")
check_outliers(online_cleaned, "Online Shoppers")
fire["frp_log"] = np.log1p(fire["frp"])
log_features = [
    "ProductRelated_Duration",
    "PageValues",
    "TotalDuration",
]
for col in log_features:
    online_cleaned[col + "_log"] = np.log1p(online_cleaned[col])

print("\n--- NASA FIRMS: LOG1P RESULT ---")
print("frp before:", round(fire["frp"].skew(), 3))
print("frp after: ", round(fire["frp_log"].skew(), 3))
print("\n--- ONLINE SHOPPERS: LOG1P RESULTS ---")
for col in log_features:
    print(
        f"{col}: {online_cleaned[col].skew():.3f}"
        f" -> {online_cleaned[col + '_log'].skew():.3f}"
    )
online_cleaned["TotalDuration_sqrt"] = np.sqrt(online_cleaned["TotalDuration"])

print("\n--- TOTALDURATION TRANSFORMATION COMPARISON ---")
print("Original:", online_cleaned["TotalDuration"].skew().round(3))
print("Log1p:", online_cleaned["TotalDuration_log"].skew().round(3))
print("Square root:", online_cleaned["TotalDuration_sqrt"].skew().round(3))
fire_scale_cols = ["frp", "brightness", "bright_t31"]
online_scale_cols = [
    "Administrative_Duration",
    "ProductRelated_Duration",
    "PageValues",
    "TotalDuration",
]

def compare_scalers(df, columns, name):
    data = df[columns].copy()
    for scaler_name, scaler in [
        ("MinMax", MinMaxScaler()),
        ("Standard", StandardScaler()),
        ("Robust", RobustScaler()),
    ]:
        scaled = scaler.fit_transform(data)
        print(f"\n{name} - {scaler_name}")
        print(
            pd.DataFrame(scaled, columns=columns)
            .describe()
            .loc[["mean", "std", "min", "50%", "max"]]
            .round(3)
        )

compare_scalers(fire, fire_scale_cols, "NASA FIRMS")
compare_scalers(online_cleaned, online_scale_cols, "Online Shoppers")
for name, df in [("NASA FIRMS", fire), ("Online Shoppers", online_cleaned)]:
    print(f"\n--- {name}: CATEGORICAL VALUES ---")
    categorical_columns = df.select_dtypes(
        include=["object", "bool", "category"]
    ).columns
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df[col].value_counts().to_dict())

online_cleaned["PageGroup"] = pd.cut(
    online_cleaned["TotalPages"],
    bins=[-1, 5, 20, float("inf")],
    labels=["Low", "Medium", "High"],
)

print("\n--- PAGEGROUP COUNTS ---")
print(online_cleaned["PageGroup"].value_counts())
print("Missing PageGroup values:", online_cleaned["PageGroup"].isna().sum())
online_cleaned["ProductAdminInteraction"] = (
    online_cleaned["ProductRelated"] * online_cleaned["Administrative"]
)

print("\n--- PRODUCT-ADMIN INTERACTION FEATURE ---")
print(
    online_cleaned[
        ["ProductRelated", "Administrative", "ProductAdminInteraction"]
    ].head()
)

print(
    "Missing values:",
    online_cleaned["ProductAdminInteraction"].isna().sum(),
)

print("Descriptive statistics:")
print(online_cleaned["ProductAdminInteraction"].describe().round(2))
fire_encoded = fire.copy()
online_encoded = online_cleaned.copy()
fire_encoded["confidence_encoded"] = fire_encoded["confidence"].map(
    {"l": 0, "n": 1, "h": 2}
)

fire_encoded = pd.get_dummies(
    fire_encoded,
    columns=["daynight", "acq_day_of_week"],
    dtype=int,
)

online_encoded = pd.get_dummies(
    online_encoded,
    columns=["Month", "VisitorType"],
    dtype=int,
)

print("\n--- NASA ENCODED COLUMNS ---")
print(
    fire_encoded.filter(
        regex="confidence_encoded|daynight_|acq_day_of_week_"
    ).head()
)

print("\n--- ONLINE SHOPPERS ENCODED COLUMNS ---")
print(online_encoded.filter(regex="Month_|VisitorType_").head())
print(
    "\nMissing confidence encodings:",
    fire_encoded["confidence_encoded"].isna().sum(),
)

print("\n--- NASA FIRMS VALIDATION ---")
print("Missing acq_month:", fire["acq_month"].isna().sum())
print("Month range:", fire["acq_month"].min(), "-", fire["acq_month"].max())
print(
    "Missing confidence encoding:",
    fire_encoded["confidence_encoded"].isna().sum(),
)

print("Confidence encoded values:")
print(sorted(fire_encoded["confidence_encoded"].dropna().unique()))
print("Missing acq_day_of_week:", fire["acq_day_of_week"].isna().sum())

print("\n--- ONLINE SHOPPERS VALIDATION ---")
print("Rows before duplicate removal:", len(online))
print("Rows after duplicate removal:", len(online_cleaned))
print("Duplicate rows remaining:", online_cleaned.duplicated().sum())
print("Missing TotalPages:", online_cleaned["TotalPages"].isna().sum())
print("Missing TotalDuration:", online_cleaned["TotalDuration"].isna().sum())
features_to_validate = [
    "TotalPages",
    "TotalDuration",
    "PageGroup",
    "ProductAdminInteraction",
]

print("Missing values in engineered features:")
print(online_cleaned[features_to_validate].isna().sum())
print("PageGroup counts:")
print(online_cleaned["PageGroup"].value_counts())
print("Rows:", len(online_cleaned))
print("\nRevenue distribution:")
print(online_cleaned["Revenue"].value_counts())
fire_hist_data = fire.drop(columns=["frp_log"], errors="ignore").select_dtypes(
    include="number"
)

online_hist_exclude = [
    "ProductRelated_Duration_log",
    "PageValues_log",
    "TotalDuration_log",
    "TotalDuration_sqrt",
    "ProductAdminInteraction",
]

online_hist_data = online_cleaned.drop(
    columns=online_hist_exclude, errors="ignore"
).select_dtypes(include="number")
fire_hist_data.hist(figsize=(15, 12), bins=30)
plt.suptitle("NASA FIRMS: Numeric Distributions")
plt.tight_layout()
plt.show()
online_hist_data.hist(figsize=(15, 12), bins=30)
plt.suptitle("Online Shoppers: Numeric Distributions")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(fire["frp"], bins=50)
axes[0, 0].set_title("FRP - Before")
axes[0, 1].hist(fire["frp_log"], bins=50)
axes[0, 1].set_title("FRP - After log1p")
axes[1, 0].hist(online_cleaned["TotalDuration"], bins=50)
axes[1, 0].set_title("TotalDuration - Before")
axes[1, 1].hist(online_cleaned["TotalDuration_log"], bins=50)
axes[1, 1].set_title("TotalDuration - After log1p")
plt.tight_layout()
plt.show()

online_cleaned[
    ["TotalDuration", "TotalDuration_log", "TotalDuration_sqrt"]
].hist(figsize=(12, 4), bins=30)
plt.tight_layout()
plt.show()

fire_final = fire.copy()
fire_final["frp_log"] = np.log1p(fire_final["frp"])

online_final = online_cleaned.copy()
online_final["TotalDuration_sqrt"] = np.sqrt(
    online_final["TotalDuration"]
)

online_final["PageValues_log"] = np.log1p(
    online_final["PageValues"]
)

fire_final.to_csv("fire_transformed.csv", index=False)
online_final.to_csv("online_transformed.csv", index=False)

print("Final datasets saved successfully.")
print("NASA FIRMS:", fire_final.shape)
print("Online Shoppers:", online_final.shape)
