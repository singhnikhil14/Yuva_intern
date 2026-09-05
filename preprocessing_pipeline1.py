
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw logistics data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df


def standardize_dates(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """
    Convert date columns to datetime type.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_cols : list
        List of column names to convert to datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized date columns.
    """
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    print("Date columns standardized.")
    return df


def standardize_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Standardize text column by stripping whitespace and title-casing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_col : str
        Column name to standardize.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized text column.
    """
    df[text_col] = df[text_col].str.strip().str.title()
    print(f"Text column '{text_col}' standardized.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate imputation strategies.

    - fuel_cost_usd: median imputation (right-skewed distribution)
    - customer_rating: mean imputation (approximately symmetric)
    - actual_arrival_date: drop rows (critical for delay analysis)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    # Median imputation for fuel_cost_usd
    median_fuel = df["fuel_cost_usd"].median()
    df["fuel_cost_usd"] = df["fuel_cost_usd"].fillna(median_fuel)
    print(f"Imputed missing fuel_cost_usd with median: {median_fuel:.2f}")

    # Mean imputation for customer_rating
    mean_rating = df["customer_rating"].mean()
    df["customer_rating"] = df["customer_rating"].fillna(mean_rating)
    print(f"Imputed missing customer_rating with mean: {mean_rating:.2f}")

    # Drop rows with missing actual_arrival_date
    missing_arrival_before = df["actual_arrival_date"].isna().sum()
    df = df.dropna(subset=["actual_arrival_date"]).reset_index(drop=True)
    print(f"Dropped {missing_arrival_before} rows with missing actual_arrival_date.")

    return df


def apply_domain_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific filters to remove unrealistic values.

    - weight_kg: must be between 0 and 50,000 kg
    - transit_time_hours: must be between 0 and 720 hours (< 30 days)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    initial_rows = len(df)

    # Filter weight
    df = df[(df["weight_kg"] > 0) & (df["weight_kg"] < 50000)]

    # Filter transit time
    df = df[(df["transit_time_hours"] > 0) & (df["transit_time_hours"] < 720)]

    df = df.reset_index(drop=True)
    print(f"Applied domain filters. Rows removed: {initial_rows - len(df)}")

    return df


def detect_and_treat_outliers(df: pd.DataFrame, col: str, 
                               lower_percentile: float = 0.05,
                               upper_percentile: float = 0.95) -> pd.DataFrame:
    """
    Detect and treat outliers by capping at specified percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to treat.
    lower_percentile : float
        Lower percentile for capping (default: 0.05).
    upper_percentile : float
        Upper percentile for capping (default: 0.95).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped.
    """
    lower_bound = df[col].quantile(lower_percentile)
    upper_bound = df[col].quantile(upper_percentile)

    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"Capped outliers in '{col}' at [{lower_bound:.2f}, {upper_bound:.2f}].")

    return df


def remove_duplicates(df: pd.DataFrame, id_col: str = "shipment_id") -> pd.DataFrame:
    """
    Remove duplicate rows and duplicate shipment IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : str
        Column name for unique identifier (default: "shipment_id").

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    initial_rows = len(df)

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Remove duplicate shipment_ids, keeping the latest by actual_dispatch_date
    df = df.sort_values("actual_dispatch_date").drop_duplicates(
        subset=[id_col], keep="last"
    )

    df = df.reset_index(drop=True)
    print(f"Removed duplicates. Rows removed: {initial_rows - len(df)}")

    return df


def fix_logical_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix logical inconsistencies such as arrival before dispatch.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with inconsistencies fixed.
    """
    # Identify records where arrival is before dispatch
    invalid_mask = df["actual_arrival_date"] < df["actual_dispatch_date"]
    invalid_count = invalid_mask.sum()
    print(f"Records with arrival before dispatch: {invalid_count}")

    # Set invalid arrival dates to NaT and drop
    df.loc[invalid_mask, "actual_arrival_date"] = pd.NaT
    df = df.dropna(subset=["actual_arrival_date"]).reset_index(drop=True)

    print(f"Dropped {invalid_count} inconsistent records.")

    return df


def normalize_features(df: pd.DataFrame, 
                       num_features: list) -> pd.DataFrame:
    """
    Apply Z-score standardization to numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    num_features : list
        List of numeric column names to standardize.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized features.
    """
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    print(f"Standardized {len(num_features)} numeric features using Z-score normalization.")

    return df


def preprocess_logistics_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    input_path : str
        Path to raw CSV file.
    output_path : str
        Path to save cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    print("=" * 60)
    print("LOGISTICS DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = load_data(input_path)

    # Step 2: Standardize dates
    print("\n[Step 2] Standardizing date columns...")
    date_cols = [
        "scheduled_dispatch_date", "actual_dispatch_date",
        "scheduled_arrival_date", "actual_arrival_date"
    ]
    df = standardize_dates(df, date_cols)

    # Step 3: Standardize text columns
    print("\n[Step 3] Standardizing text columns...")
    df = standardize_text(df, "carrier")

    # Convert categorical columns to category type
    cat_cols = ["origin_city", "destination_city", "carrier", "status"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    print(f"Converted {len(cat_cols)} columns to category type.")

    # Step 4: Handle missing values
    print("\n[Step 4] Handling missing values...")
    df = handle_missing_values(df)

    # Step 5: Apply domain filters
    print("\n[Step 5] Applying domain filters...")
    df = apply_domain_filters(df)

    # Step 6: Detect and treat outliers
    print("\n[Step 6] Detecting and treating outliers...")
    df = detect_and_treat_outliers(df, "delay_hours", 0.05, 0.95)

    # Step 7: Remove duplicates
    print("\n[Step 7] Removing duplicates...")
    df = remove_duplicates(df, "shipment_id")

    # Step 8: Fix logical inconsistencies
    print("\n[Step 8] Fixing logical inconsistencies...")
    df = fix_logical_inconsistencies(df)

    # Step 9: Normalize numeric features
    print("\n[Step 9] Normalizing numeric features...")
    num_features = [
        "weight_kg", "volume_m3", "distance_km",
        "transit_time_hours", "fuel_cost_usd", "delay_hours"
    ]
    df = normalize_features(df, num_features)

    # Step 10: Save cleaned data
    print("\n[Step 10] Saving cleaned data...")
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFinal dataset shape: {df.shape}")
    print("\nMissing values after cleaning:")
    print(df.isna().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nStandardized features summary:")
    print(df[num_features].describe())

    return df


if __name__ == "__main__":
    # File paths
    INPUT_FILE = "shipping_operations_raw.csv"
    OUTPUT_FILE = "shipping_operations_cleaned.csv"

    # Run preprocessing pipeline
    cleaned_df = preprocess_logistics_data(INPUT_FILE, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("Pipeline execution finished successfully.")
    print("=" * 60)
