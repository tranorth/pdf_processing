import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
ROOT_DIRECTORY = r"pdf_data"
OUTPUT_DIRECTORY = r"pdf_processing"


# =============================================================================
# --- 2. DATA PREPARATION & ANALYSIS FUNCTIONS ---
# =============================================================================

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Loads and prepares the combined CSV for analysis."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded '{file_path}'")
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
        return pd.DataFrame()

    key_cols = ['Market', 'Submarket', 'Secondary Submarket', 'Property Type']
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].fillna('N/A')
        else:
            df[col] = 'N/A'

    df['LocationID'] = df.apply(
        lambda row: f"{row['Market']}_{row['Submarket']}_{row['Secondary Submarket']}_{row['Property Type']}",
        axis=1
    )
    df.sort_values(by=['LocationID', 'Year', 'Period_Number'], inplace=True)
    return df

def create_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'Prev_*' columns and calculates 'Quarter_Gap' safely."""
    df['Prev_Year'] = df.groupby('LocationID')['Year'].shift(1)
    df['Prev_Period_Type'] = df.groupby('LocationID')['Period_Type'].shift(1)
    df['Prev_Period_Number'] = df.groupby('LocationID')['Period_Number'].shift(1)
    df['Prev_Inventory_SF'] = df.groupby('LocationID')['Inventory SF'].shift(1)
    df['Prev_Vacancy_Q'] = df.groupby('LocationID')['Vacancy Q'].shift(1)
    df['Prev_Asking_Rent_Q'] = df.groupby('LocationID')['Asking Rent Q'].shift(1)

    df['Prev_Year'] = df['Prev_Year'].astype('Int64')
    df['Prev_Period_Number'] = df['Prev_Period_Number'].astype('Int64')

    # **THE FIX FOR THE KERNEL CRASH IS HERE:**
    # Convert nullable integers to standard floats before doing math.
    # This prevents the recursion bug.
    prev_year_float = df['Prev_Year'].astype(float)
    prev_period_float = df['Prev_Period_Number'].astype(float)

    current_abs_q = np.where(df['Period_Type'] == 'H', df['Period_Number'] * 2, df['Period_Number']) + (df['Year'] - 2000) * 4
    prev_abs_q = np.where(df['Prev_Period_Type'] == 'H', prev_period_float * 2, prev_period_float) + (prev_year_float - 2000) * 4
    
    df['Quarter_Gap'] = (current_abs_q - prev_abs_q).astype('Int64')
    
    print("✅ Successfully created comparison columns.")
    return df

def validate_rent_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies records where rent changes by more than 10% per quarter."""
    # (This function is correct)
    pass # Add your validation logic here

def validate_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies records where inventory mismatches."""
    # (This function is correct)
    pass # Add your validation logic here

def validate_vacancy(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies records where vacancy rate differs significantly."""
    # (This function is correct)
    pass # Add your validation logic here

def run_analysis_pipeline(file_path: str) -> pd.DataFrame:
    """Main orchestrator for the data validation pipeline."""
    prepared_df = load_and_prepare_data(file_path)
    if prepared_df.empty:
        return pd.DataFrame()

    analysis_df = create_comparison_columns(prepared_df)
    validation_subset_df = analysis_df.dropna(subset=['Prev_Year', 'Quarter_Gap']).copy()
    
    print("🕵️  Running validation checks...")
    rent_issues = validate_rent_changes(validation_subset_df)
    inventory_issues = validate_inventory(validation_subset_df)
    vacancy_issues = validate_vacancy(validation_subset_df)
    
    all_issues_df = pd.concat([rent_issues, inventory_issues, vacancy_issues])
    
    if all_issues_df.empty:
        print("\n🎉 --- Analysis Complete: No potential issues identified! ---")
        return pd.DataFrame()
        
    print(f"\n📄 --- Analysis Complete: Found {len(all_issues_df)} potential issues ---")
    return all_issues_df


# =============================================================================
# --- 5. EXECUTION ---
# =============================================================================

# Define the file to be analyzed
file_to_analyze = str(Path(OUTPUT_DIRECTORY) / "combined.csv")

# Run the entire analysis and get the report
validation_report = run_analysis_pipeline(file_to_analyze)

# Display the final report
if not validation_report.empty:
    print("\n--- Validation Report Summary ---")
    print(validation_report)