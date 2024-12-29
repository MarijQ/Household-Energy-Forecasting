import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd

# ========================================
# Global Settings and Configuration
# ========================================

# Library settings
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

# Directories
CHARTS_DIR = "./Marij/charts"
TEST_DATA_DIR = "./Marij/ind_homes"  # Directory for independent home data
os.makedirs(CHARTS_DIR, exist_ok=True)

def clear_chart_directory():
    """Clear all files in the charts directory."""
    for filename in os.listdir(CHARTS_DIR):
        file_path = os.path.join(CHARTS_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# ========================================
# Data Preprocessing Functions
# ========================================


def rename_columns(column_name):
    """Rename columns for simplicity."""
    parts = column_name.split('_')
    if len(parts) > 1:
        room = parts[1][:3]
        sensor_map = {
            'light': 'light',
            'humidity': 'humid',
            'temperature': 'temp',
            'hot-water-cold-pipe': 'coldpipe',
            'hot-water-hot-pipe': 'hotpipe',
            'central-heating-return': 'chreturn',
            'central-heating-flow': 'chflow',
            'electric-combined': 'elec',
            'gas': 'gas'
        }
        sensor = sensor_map.get(parts[-1], parts[-1])
        return f"{room}_{sensor}"
    return column_name


def preprocess_data(df):
    """
    Preprocess the dataset: rename columns, handle missing values,
    enforce time index frequency, and derive electricity and gas features.
    """
    # Rename columns for simplicity
    df.rename(columns={col: rename_columns(col)
              for col in df.columns}, inplace=True)

    # Ensure 'timestamp' is in datetime format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Enforce a consistent frequency for the time index
    start_time = df.index.min()
    end_time = df.index.max()
    full_index = pd.date_range(
        start=start_time, end=end_time, freq='h')  # Hourly frequency

    # Reindex the DataFrame and fill missing timestamps with NaNs
    df = df.reindex(full_index)
    df.index.name = 'timestamp'

    # Impute missing values for key columns
    if 'hal_elec' in df.columns:
        df['hal_elec'] = df['hal_elec'].interpolate(
            method='linear').bfill().ffill()
    if 'hal_gas' in df.columns:
        df['hal_gas'] = df['hal_gas'].fillna(0)

    # Convert and sum electricity and gas data
    df['elec'] = df.get('hal_elec', 0) * 0.00027778  # Convert joules to Wh
    df['gas'] = df.get('hal_gas', 0)  # Already in Wh

    # Retain only relevant columns
    df = df[['elec', 'gas']].dropna()
    return df


# ========================================
# Time Series Analysis and Modeling
# ========================================


def fit_arima_and_save_charts(df, col, order, home_name):
    """
    Fit ARIMA model, save fitted vs actual and residual plots for the specified column.
    """
    # Fit ARIMA model
    model = ARIMA(df[col], order=order)
    fitted_model = model.fit()
    df['fitted'] = fitted_model.fittedvalues

    # Fitted vs Actual chart
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[col], label='Actual', color='blue')
    plt.plot(df.index, df['fitted'], label='Fitted', color='orange')
    plt.title(
        f'{home_name} - {col.capitalize()} - Fitted vs Actual', fontsize=16)
    plt.legend()
    fitted_chart_path = os.path.join(
        CHARTS_DIR, f"{home_name}_{col}_fitted_vs_actual.png")
    plt.savefig(fitted_chart_path)
    plt.close()
    # print(f"Saved chart: {fitted_chart_path}")

    # Residuals chart
    residuals = df[col] - df['fitted']
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuals', color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{home_name} - {col.capitalize()} - Residuals', fontsize=16)
    residuals_chart_path = os.path.join(
        CHARTS_DIR, f"{home_name}_{col}_residuals.png")
    plt.savefig(residuals_chart_path)
    plt.close()
    # print(f"Saved chart: {residuals_chart_path}")

    return fitted_model


def check_stationarity(series, series_name):
    """Perform the Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series)
    print(f"ADF Statistic for {series_name}: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print(f"The {series_name} series is stationary.")
    else:
        print(f"The {series_name} series is not stationary.")


# ========================================
# Evaluation and Cross-Validation
# ========================================

def evaluate_model(df, col, fitted_col):
    """Evaluate the model and calculate Normalized RMSE."""
    rmse = np.sqrt(mean_squared_error(df[col], df[fitted_col]))
    mean_val = df[col].mean()
    nrmse = rmse / mean_val
    return nrmse


def cross_validate_model(test_files, base_model_order, columns):
    """
    Perform cross-validation on a random subset of test files for specified columns.
    """
    results = {}
    selected_files = random.sample(test_files, 5)  # Randomly choose 5 files

    # Process each selected file
    for file in selected_files:
        # Use file name (e.g., home_home59) as home name
        home_name = file.split('.')[0]
        print(f"Processing test file: {file}")

        # Load and preprocess the file
        test_df = pd.read_csv(os.path.join(TEST_DATA_DIR, file))
        test_df = preprocess_data(test_df)
        test_df.index = pd.date_range(
            start=test_df.index.min(), periods=len(test_df), freq='h')

        # Store accuracy metrics for both columns (e.g., 'elec' and 'gas')
        home_results = {}
        for col in columns:
            print(f"Fitting ARIMA model for {col}...")
            fit_arima_and_save_charts(
                test_df, col, base_model_order, home_name)  # Save charts
            nrmse = evaluate_model(test_df, col, 'fitted')  # Evaluate model
            home_results[col] = nrmse

        results[home_name] = home_results
    return results


# ========================================
# Main Execution
# ========================================

def main():
    clear_chart_directory()
    data_file = './sensor_data_47.csv'
    base_model_order = (1, 0, 1)
    evaluation_columns = ['elec', 'gas']  # Columns to evaluate separately

    # Load and preprocess the primary dataset
    df = pd.read_csv(data_file)
    df = preprocess_data(df)

    # Stationarity check on electricity consumption
    print("Performing stationarity check on electricity...")
    check_stationarity(df['elec'], 'Electricity Consumption')

    # Fit and save ARIMA charts for primary dataset
    print("Fitting ARIMA for primary dataset...")
    for col in evaluation_columns:
        fitted_model = fit_arima_and_save_charts(df, col, base_model_order, 'Primary_Dataset')
        
        # Calculate and print RMSE for the primary dataset
        nrmse = evaluate_model(df, col, 'fitted')
        print(f"Normalized RMSE for {col.capitalize()} in Primary Dataset: {nrmse:.4f}")

    # Evaluate on random test homes
    print("Starting cross-validation...")
    test_files = os.listdir(TEST_DATA_DIR)
    cross_val_results = cross_validate_model(
        test_files, base_model_order, evaluation_columns)

    # Output cross-validation results
    print("Cross-validation results (Normalized RMSE per home per column):")
    for home, metrics in cross_val_results.items():
        print(f"{home}:")
        for col, nrmse in metrics.items():
            print(f"  {col.capitalize()}: {nrmse:.4f}")


def test():
    data_file = './Marij/ind_homes/home_home167.csv'
    base_model_order = (1, 0, 1)
    evaluation_columns = ['elec', 'gas']  # Columns to evaluate separately

    # Load and preprocess the primary dataset
    df = pd.read_csv(data_file)
    df = preprocess_data(df)
    df.set_index('timestamp', inplace=True)

    # Fit and save ARIMA charts for primary dataset
    print("Fitting ARIMA for primary dataset...")
    for col in evaluation_columns:
        fit_arima_and_save_charts(df, col, base_model_order, 'Primary_Dataset')


if __name__ == "__main__":
    main()
