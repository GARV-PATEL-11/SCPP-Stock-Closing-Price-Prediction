

# === Import Libraries ===
import pandas as pd

# === Data Cleaning Function ===
def drop_all_null_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes columns with all null values and drops rows with any null values.

    Parameters:
    ----------
    df : pd.DataFrame
        The input pandas DataFrame to be cleaned.

    Returns:
    -------
    tuple:
        - pd.DataFrame: Cleaned DataFrame with null columns and rows removed.
        - list[str]: Names of columns that were dropped.

    Raises:
    ------
    ValueError:
        If the input is not a pandas DataFrame.

    Notes:
    -----
    - This function does not modify the input DataFrame in-place.
    - Columns where all values are NaN will be dropped.
    - Rows containing any NaN value will also be removed.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Identify columns with all null values
    all_null_columns = df.columns[df.isna().all()].tolist()

    # Drop all-null columns, then drop rows with any nulls
    df_clean = df.drop(columns=all_null_columns).dropna()

    # Display dropped columns
    if all_null_columns:
        print(f"Dropped {len(all_null_columns)} column(s) with all null values:")
        for col in all_null_columns:
            print(f" - {col}")
    else:
        print("No columns with all null values found.")

    return df_clean, all_null_columns
