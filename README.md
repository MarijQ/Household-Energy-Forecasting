# ML-Project

# 1. Intoduction
The goal of our project is to predict how much energy people might use in the future for electricity, gas, and water. By looking at past energy usage, weather data, and other factors, we want to help people better understand their energy habits. This can make it easier to plan, save money, and use resources more wisely. In this project, we’ve worked as a team to build a tool that uses data and machine learning to make these predictions. 

--- Intorduce the team work and what was the tasks for all the team member

# 2. Key Features

### ***Ml Models:***

We explored and implemented several models to predict energy consumption:

- ARIMA and SARIMA: For capturing patterns and seasonality in the energy usage data.
 
- LSTM: A deep learning approach that helps uncover complex patterns and trends over time.
 
- Prophet: A model well-suited for time series data, helping us make reliable and interpretable predictions.
 

### ***Tech Stack***:

The project leverages Python as the primary programming language, alongside essential libraries such as Pandas, NumPy, Matplotlib, TensorFlow, and Facebook’s Prophet. These tools allowed us to preprocess data, train models, and visualize results effectively.

--remove this-- model used, teck stack used, dataset used

# 3. Data Prep

The IDEAL Household Energy Dataset provided a wealth of information on energy usage across 255 homes, including electricity, gas, and water consumption, as well as metadata on household characteristics, sensors, and weather conditions. Preparing this data was essential to ensure its accuracy and usability for our project.

### ***Data Source***:

The dataset was sourced from the [IDEAL Household Energy Dataset](https://datashare.ed.ac.uk/handle/10283/3647). It includes time-series data from sensors installed in homes, along with metadata such as home layouts, appliance types, and environmental factors.

### ***Data Cleaning***:

- For most features, missing values were handled using forward and backward filling, ensuring continuity in time-series data.
- Gas data had the highest proportion of missing values. In these cases, missing values were replaced with 0, assuming no gas usage during those periods, as outlined in the dataset documentation.
- Anomalous readings, such as unrealistically high temperatures or humidity levels, were identified and removed based on the dataset guidelines.

### ***Feature Engineering***:

To enhance the dataset for modeling, additional features were created:

- Time-Based Features: Indicators for weekdays, weekends, and public holidays to capture patterns in energy usage.
- Weather Data Integration: Features like temperature, humidity, and wind speed were added, as weather has a direct impact on energy consumption.
- Lag Features: Historical energy usage values (e.g., 1-day and 7-day lags) were introduced to capture trends over time.

--remove this-- data cleaning, feature engineering, table of features used. 

# 4. ML Models

In this project, we used several machine learning models, each chosen for their ability to handle time series data and make accurate predictions about energy usage. Here’s a brief overview of the models used and why we chose them:

### ***ARIMA (AutoRegressive Integrated Moving Average)***:

ARIMA is a statistical model that helps us understand and predict future values based on past data. It’s useful for time series data like energy consumption, where the past behavior of the data can help us forecast future trends.

### ***SARIMA (Seasonal ARIMA)***:

SARIMA builds on ARIMA but includes the ability to handle seasonal patterns in the data. Since energy usage can often be affected by seasonal factors (like weather), SARIMA helps capture these regular fluctuations to improve predictions.

### ***LSTM (Long Short-Term Memory)***:

LSTM is a type of deep learning model designed for sequential data. It’s particularly good at learning long-term dependencies and capturing complex patterns over time. This model is used to detect intricate patterns in energy usage that might not be easily visible with traditional methods.

### ***Prophet***:
 
Prophet is an open-source model developed by Facebook for forecasting time series data. It’s easy to use and works well with data that has strong seasonal effects, like energy consumption. We used it because it’s reliable and interpretable, which makes it a good choice for predicting future energy use.

--remove this-- brief definition about all the models used and why is it used.

# 5. Results

The results of our energy consumption prediction models highlight the performance of each approach for forecasting electricity and gas usage. Below is a summary of the evaluation results using RMSE (Root Mean Square Error) as the metric:

### Evaluation Results (RMSE):

### ***Electricity Consumption***:

- ARIMA: 4579.267
- SARIMAX: 1231.687
- LSTM Single: 990.731
- LSTM Sequential: 1178.299
- Prophet: 2482.880
- Gas Consumption:
- ARIMA: 2936.300
- SARIMAX: 2937.412
- LSTM Single: 5266.041
- LSTM Sequential: 2984.296
- Prophet: 12197.262
  
### Key Observations:

### ***Electricity Predictions***:

LSTM Single performed the best, achieving the lowest RMSE, followed by SARIMAX and LSTM Sequential. Prophet and ARIMA were less effective for electricity consumption predictions.

### ***Gas Predictions***:

ARIMA performed better compared to the other models, though its RMSE was still relatively high. Prophet struggled significantly with gas data, likely due to the higher percentage of missing values and imputation assumptions.

### ***Insights Gained***:

- Electricity consumption was easier to predict, showing more consistent patterns across the dataset.
- Gas consumption predictions were challenging, as the dataset had significant missing values and less consistent patterns.
-	Weather conditions and time-based features had a noticeable impact on both electricity and gas usage trends.

These results highlight the need for choosing models based on the characteristics of the dataset and the specific energy type being analyzed.

# 6. Running the code (script)
explain how to run the code, dependencies and requirements

# 7. Furture Work
what else can be done, new methods (if any), potential improvements in the current work.

# 8. Licence
open source. anyone can contrbute
