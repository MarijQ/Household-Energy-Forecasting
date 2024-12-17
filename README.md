# ML-Project

# 1. Intoduction

# 2. Key Features

# 3. Data Prep

# 4. ML Models

# 5. Results

- **ARIMA Model**:
The ARIMA model did a decent job at predicting both electricity and gas usage. However, it struggled more with gas data because it's a simpler model that can miss complex patterns.

- **SARIMAX Model**:
SARIMAX was excellent at predicting electricity usage since it can use additional variables to improve accuracy. However, it didn’t work well for gas predictions, likely because gas patterns are more complex.

- **LSTM Single Layer**:
The single-layer LSTM didn’t perform well for either electricity or gas predictions. It likely overfit the training data, meaning it learned too much from the training set and couldn’t handle new data effectively.

- **LSTM Sequential**:
The sequential LSTM did great for electricity predictions because it’s good at learning patterns over time. However, it struggled with gas data, likely because gas usage patterns are less predictable and harder to model.

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
