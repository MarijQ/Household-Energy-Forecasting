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

### Analysis:

1.	Best-Performing Model:

LSTM (Single) had the lowest RMSE (990.731), making it the most accurate model for predicting electricity consumption. Its ability to capture long-term dependencies and trends in sequential data contributed to its strong performance.

2.	SARIMAX and LSTM (Sequential):

- SARIMAX followed closely with an RMSE of 1231.687, benefiting from its ability to handle seasonality in the electricity data.
- LSTM (Sequential) also performed well (RMSE: 1178.299), though it slightly lagged behind its Single variant due to the complexity of training sequential layers effectively.

3.	Moderate Performance of Prophet:

Prophet achieved an RMSE of 2482.880. While it captured seasonal trends effectively, it struggled with more complex, non-linear relationships in the data.

4.	Least Effective Model:

ARIMA had the highest RMSE (4579.267) among electricity models. Its limitation in handling seasonal and trend variations in larger datasets led to lower accuracy compared to other models.

### ***Insights***:

- Electricity consumption exhibited relatively stable patterns, making it easier to predict.
- Time-based features such as weekends and holidays significantly influenced electricity usage trends.
- The use of lag features and weather data further enhanced the performance of advanced models like LSTM.
 
### ***Gas Consumption***:

- ARIMA: 2936.300
- SARIMAX: 2937.412
- LSTM Single: 5266.041
- LSTM Sequential: 2984.296
- Prophet: 12197.262

### ***Analysis***:

1. *Best-Performing Model*:

ARIMA produced the lowest RMSE (2936.300), showing that its simple, statistical approach was well-suited for gas consumption data, especially when dealing with short-term dependencies.

2.	*Similar Performance of SARIMAX*:

SARIMAX closely followed ARIMA with an RMSE of 2937.412. However, the inclusion of seasonal components didn’t offer significant improvements, possibly due to the inconsistent patterns in gas data.

3.	*LSTM Models*:

- LSTM (Sequential) performed moderately (RMSE: 2984.296), slightly worse than ARIMA and SARIMAX. Its results suggest that while LSTMs can capture complex patterns, they struggled due to the high proportion of missing values and irregularities in gas usage.
- LSTM (Single) had the poorest performance among the models (RMSE: 5266.041). It indicates that the Single LSTM struggled to generalize well on this dataset.

4.	*Poor Performance of Prophet*:

Prophet showed the highest RMSE (12197.262), indicating that it struggled to handle the irregular and sparse nature of the gas consumption data.

### ***Insights***:

- Gas consumption predictions were more challenging due to missing data and irregular patterns.
- Imputation of missing values (with 0) likely affected the models’ ability to accurately capture trends in gas usage.
- Weather features, particularly temperature, had a significant influence on gas usage, with higher consumption during colder periods.
  
### ***Overall Observations***:
- Electricity predictions were generally more accurate due to stable and consistent patterns in the data. Advanced models like LSTM Single and SARIMAX performed the best.
- Gas predictions faced challenges due to irregularities and data gaps, with simpler models like ARIMA outperforming more complex methods.

These results highlight the importance of tailoring model selection to the characteristics of the dataset and the type of energy being predicted.

# 6. Running the code (script)
explain how to run the code, dependencies and requirements

# 7. Furture Work
what else can be done, new methods (if any), potential improvements in the current work.

# 8. Licence
open source. anyone can contrbute
