# TODO
# ARIMA hyperparameter tuning (or auto-tune)
# SARIMAX
# Fix cross-validation (cleaning of other files)

import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler


# ========================================
# Global Settings and Configuration
# ========================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# TODO
# ARIMA hyperparameter tuning (or auto-tune)
# SARIMAX
# Fix cross-validation (cleaning of other files)

import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler


# ========================================
# Global Settings and Configuration
# ========================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Library settings
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

# Directories
CHARTS_DIR = r"./Marij/charts"
TEST_DATA_DIR = r"./ind-homes-clean"  # Directory for independent home data
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
    parts = column_name.split("_")
    if len(parts) > 1:
        room = parts[1][:3]
        sensor_map = {
            "light": "light",
            "humidity": "humid",
            "temperature": "temp",
            "hot-water-cold-pipe": "coldpipe",
            "hot-water-hot-pipe": "hotpipe",
            "central-heating-return": "chreturn",
            "central-heating-flow": "chflow",
            "electric-combined": "elec",
            "gas": "gas",
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
    df.rename(columns={col: rename_columns(col) for col in df.columns}, inplace=True)

    # Ensure 'timestamp' is in datetime format
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    # Enforce a consistent frequency for the time index
    start_time = df.index.min()
    end_time = df.index.max()
    full_index = pd.date_range(start=start_time, end=end_time, freq="h")  # Hourly

    # Reindex the DataFrame and fill missing timestamps with NaNs
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    # Impute missing values for key columns
    if "hal_elec" in df.columns:
        df["hal_elec"] = df["hal_elec"].interpolate(method="linear").bfill().ffill()
    if "hal_gas" in df.columns:
        df["hal_gas"] = df["hal_gas"].fillna(0)

    # Convert and sum electricity and gas data
    df["elec"] = df.get("hal_elec", 0) * 0.00027778  # Convert joules to Wh
    df["gas"] = df.get("hal_gas", 0)  # Already in Wh

    # Retain only relevant columns
    df = df[["elec", "gas"]].dropna()
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
    df["fitted"] = fitted_model.fittedvalues

    # Fitted vs Actual chart
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[col], label="Actual", color="blue")
    plt.plot(df.index, df["fitted"], label="Fitted", color="orange")
    plt.title(f"{home_name} - {col.capitalize()} - Fitted vs Actual", fontsize=16)
    plt.legend()
    fitted_chart_path = os.path.join(
        CHARTS_DIR, f"{home_name}_{col}_fitted_vs_actual.png"
    )
    plt.savefig(fitted_chart_path)
    plt.close()
    # print(f"Saved chart: {fitted_chart_path}")

    # Residuals chart
    residuals = df[col] - df["fitted"]
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label="Residuals", color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{home_name} - {col.capitalize()} - Residuals", fontsize=16)
    residuals_chart_path = os.path.join(CHARTS_DIR, f"{home_name}_{col}_residuals.png")
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


def fit_lstm_and_save_charts(df, col, home_name):
    """
    Fit an LSTM model, save fitted vs actual and residual plots for the specified column.
    """
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[col].values.reshape(-1, 1))

    # Prepare the data for LSTM
    sequence_length = 24  # Use past 24 hours to predict the next value
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    # [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    time_index = df.index[sequence_length:]  # Adjust index for the sequences

    # Build the LSTM model
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Prepare DataFrame for plotting
    df_pred = pd.DataFrame(
        {
            "timestamp": time_index,
            "actual": df[col].values[sequence_length:],
            "predicted": predictions.flatten(),
        }
    ).set_index("timestamp")

    # Fitted vs Actual chart
    plt.figure(figsize=(10, 6))
    plt.plot(df_pred.index, df_pred["actual"], label="Actual", color="blue")
    plt.plot(df_pred.index, df_pred["predicted"], label="Predicted", color="green")
    plt.title(f"{home_name} - {col.capitalize()} - LSTM Fitted vs Actual", fontsize=16)
    plt.legend()
    fitted_chart_path = os.path.join(
        CHARTS_DIR, f"{home_name}_{col}_lstm_fitted_vs_actual.png"
    )
    plt.savefig(fitted_chart_path)
    plt.close()

    # Residuals chart
    residuals = df_pred["actual"] - df_pred["predicted"]
    plt.figure(figsize=(10, 6))
    plt.plot(residuals.index, residuals, label="Residuals", color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{home_name} - {col.capitalize()} - LSTM Residuals", fontsize=16)
    residuals_chart_path = os.path.join(
        CHARTS_DIR, f"{home_name}_{col}_lstm_residuals.png"
    )
    plt.savefig(residuals_chart_path)
    plt.close()

    # Calculate Normalized RMSE
    rmse = np.sqrt(mean_squared_error(df_pred["actual"], df_pred["predicted"]))
    mean_val = df_pred["actual"].mean()
    nrmse = rmse / mean_val

    print(
        f"Normalized RMSE for {col.capitalize()} using LSTM in {home_name}: {nrmse:.4f}"
    )


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
        home_name = file.split(".")[0]
        print(f"Processing test file: {file}")

        # Load and preprocess the file
        test_df = pd.read_csv(os.path.join(TEST_DATA_DIR, file))
        test_df = preprocess_data(test_df)
        test_df.index = pd.date_range(
            start=test_df.index.min(), periods=len(test_df), freq="h"
        )

        # Store accuracy metrics for both columns (e.g., 'elec' and 'gas')
        home_results = {}
        for col in columns:
            print(f"Fitting ARIMA model for {col}...")
            fit_arima_and_save_charts(
                test_df, col, base_model_order, home_name
            )  # Save charts
            nrmse = evaluate_model(test_df, col, "fitted")  # Evaluate model
            home_results[col] = nrmse

        results[home_name] = home_results
    return results


# ========================================
# Main Execution
# ========================================


def main():
    clear_chart_directory()
    data_file = r"./sensor_data_47.csv"
    base_model_order = (1, 0, 1)
    evaluation_columns = ["elec", "gas"]  # Columns to evaluate separately

    # Load and preprocess the primary dataset
    df = pd.read_csv(data_file)
    df = preprocess_data(df)

    # Stationarity check on electricity consumption
    print("Performing stationarity check on electricity...")
    check_stationarity(df["elec"], "Electricity Consumption")

    # Fit and save ARIMA charts for primary dataset
    print("Fitting ARIMA for primary dataset...")
    for col in evaluation_columns:
        fitted_model = fit_arima_and_save_charts(
            df, col, base_model_order, "Primary_Dataset"
        )

        # Calculate and print NRMSE for the primary dataset using ARIMA
        nrmse_arima = evaluate_model(df, col, "fitted")
        print(
            f"Normalized RMSE for {col.capitalize()} using ARIMA in Primary Dataset: {nrmse_arima:.4f}"
        )

    # Fit and save LSTM charts for primary dataset
    print("Fitting LSTM for primary dataset...")
    for col in evaluation_columns:
        fit_lstm_and_save_charts(df, col, "Primary_Dataset")

    # Cross-validation code can be commented out for now
    print("Starting cross-validation...")
    test_files = os.listdir(TEST_DATA_DIR)
    cross_val_results = cross_validate_model(
        test_files, base_model_order, evaluation_columns
    )

    # Output cross-validation results
    print("Cross-validation results (Normalized RMSE per home per column):")
    for home, metrics in cross_val_results.items():
        print(f"{home}:")
        for col, nrmse in metrics.items():
            print(f"  {col.capitalize()}: {nrmse:.4f}")

if __name__ == "__main__":
    main()