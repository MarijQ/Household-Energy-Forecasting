# ML-Project

# 1. Intoduction

# 2. Key Features

# 3. Data Prep

# 4. ML Models

# 5. Results

- **ARIMA Model**:
This model performed reasonably well for both electricity and gas predictions. However, it struggled more with gas consumption patterns, likely because ARIMA's simplicity makes it less effective for capturing complex dynamics.

- **SARIMAX Model**:
SARIMAX excelled in predicting electricity consumption, leveraging its ability to incorporate exogenous variables. However, it failed to perform well for gas predictions, suggesting the need for a more sophisticated model for gas data.

- **LSTM Single Layer**:
The single-layer LSTM model performed poorly overall, especially for gas consumption. This indicates that the model might have overfitted the training data and failed to generalize well to unseen data.

- **LSTM Sequential**:
The sequential LSTM model showed strong results for electricity predictions, likely due to its ability to learn sequential dependencies effectively. However, it performed significantly worse for gas predictions, possibly due to the diverse and less sequential nature of gas consumption patterns.

# Context

# Project Scope: 

Develop an ML model to predict gas, water and electricity consumption based on historical data, weather data and other features related to the subject area. The main aim is to predict accurate energy consumption for proper future planning. 

Since the energy consumption for all three types is not possible due to different unit measurements, predicting separately provides accurate information. 

# Data Required: 

- Historical Data (energy Consumption)
- Weather data
- Temperature data.
- Datetime
- Other household data

***NOTE***: Here, two datasets are used, weather forecasting data and hourly readings for each household along with the specific house area readings such as kitchen, hall, leaving room etc.

## Approach: 

- Data Preprocessing
- Feature Engineering
- Modeling
