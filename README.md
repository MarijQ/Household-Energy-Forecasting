# ML-Project  

---

> ### Note: Archived Repository
> This repository represents the completed version of the project (as of 29/12/2024) collaboratively built by:
> 
> [Marij Qureshi](https://github.com/MarijQ), [Georgios Gkakos](https://github.com/GGkakos), [Het Suhagiya](https://github.com/HetSuhagiya)
> 
> This repository is no longer actively maintained. Please see individual forks for the most up-to-date versions.

---

## 1. Introduction  

The goal of our project is to predict how much energy people might use in the future for electricity, gas, and water. By looking at past energy usage, weather data, and other factors, we want to help people better understand their energy habits. This can make it easier to plan, save money, and use resources more wisely.  

In this project, we’ve worked as a team to build a tool that uses data and machine learning to make these predictions.  

### ***Team Contributions:***  
Each team member contributed to different aspects of the project:  
- **Data Preparation:** Cleaning, preprocessing, and feature engineering.  
- **Model Implementation:** Experimenting with and fine-tuning machine learning models.  
- **Evaluation and Analysis:** Interpreting results and refining predictions.  
- **Visualization and Documentation:** Creating visualizations and maintaining the project documentation.  

---

## 2. Key Features  

### ***ML Models:***  

We explored and implemented several models to predict energy consumption:  
- **ARIMA and SARIMA:** For capturing patterns and seasonality in the energy usage data.  
- **LSTM:** A deep learning approach that helps uncover complex patterns and trends over time.  
- **Prophet:** A model well-suited for time series data, helping us make reliable and interpretable predictions.  

### ***Tech Stack:***  

The project leverages Python as the primary programming language, alongside essential libraries such as Pandas, NumPy, Matplotlib, TensorFlow, and Facebook’s Prophet. These tools allowed us to preprocess data, train models, and visualize results effectively.  

---

## 3. Data Prep  

The IDEAL Household Energy Dataset provided a wealth of information on energy usage across 255 homes, including electricity, gas, and water consumption, as well as metadata on household characteristics, sensors, and weather conditions. Preparing this data was essential to ensure its accuracy and usability for our project.  

### ***Data Source:***  

The dataset was sourced from the [IDEAL Household Energy Dataset](https://datashare.ed.ac.uk/handle/10283/3647). It includes time-series data from sensors installed in homes, along with metadata such as home layouts, appliance types, and environmental factors.  

### ***Data Cleaning:***  

- For most features, missing values were handled using forward and backward filling, ensuring continuity in time-series data.  
- Gas data had the highest proportion of missing values. In these cases, missing values were replaced with 0, assuming no gas usage during those periods, as outlined in the dataset documentation.  
- Anomalous readings, such as unrealistically high temperatures or humidity levels, were identified and removed based on the dataset guidelines.  

### ***Feature Engineering:***  

To enhance the dataset for modeling, additional features were created:  
- **Time-Based Features:** Indicators for weekdays, weekends, and public holidays to capture patterns in energy usage.  
- **Weather Data Integration:** Features like temperature, humidity, and wind speed were added, as weather has a direct impact on energy consumption.  
- **Lag Features:** Historical energy usage values (e.g., 1-day and 7-day lags) were introduced to capture trends over time.  

---

## 4. ML Models  

In this project, we used several machine learning models, each chosen for their ability to handle time series data and make accurate predictions about energy usage.  

### ***ARIMA (AutoRegressive Integrated Moving Average):***  

ARIMA is a statistical model that helps us understand and predict future values based on past data. It’s useful for time series data like energy consumption, where the past behavior of the data can help us forecast future trends.  

### ***SARIMA (Seasonal ARIMA):***  

SARIMA builds on ARIMA but includes the ability to handle seasonal patterns in the data. Since energy usage can often be affected by seasonal factors (like weather), SARIMA helps capture these regular fluctuations to improve predictions.  

### ***LSTM (Long Short-Term Memory):***  

LSTM is a type of deep learning model designed for sequential data. It’s particularly good at learning long-term dependencies and capturing complex patterns over time. This model is used to detect intricate patterns in energy usage that might not be easily visible with traditional methods.  

### ***Prophet:***  

Prophet is an open-source model developed by Facebook for forecasting time series data. It’s easy to use and works well with data that has strong seasonal effects, like energy consumption. We used it because it’s reliable and interpretable, which makes it a good choice for predicting future energy use.  

---

## 5. Results  

The results of our energy consumption prediction models highlight the performance of each approach for forecasting electricity and gas usage. Below is a summary of the evaluation results using RMSE (Root Mean Square Error) as the metric:  

### ***Evaluation Results (RMSE):***  

#### **Electricity Consumption:**  
| Model             | RMSE       |
|-------------------|------------|
| ARIMA             | 4579       |
| SARIMAX           | 1231       |
| LSTM Single       | 990        |
| LSTM Sequential   | 1178       |
| Prophet           | 2482       |

**Electricity Actual vs Fitted**  
![image](https://github.com/user-attachments/assets/3a4e3498-8b8a-465d-af0d-2e40b150f5a6)

#### ***Analysis:***  

- **Best-Performing Model:**  
  Our LSTM (Single) model achieved the lowest RMSE of 990, making it the most accurate for predicting electricity consumption. Its ability to capture long-term dependencies and trends in sequential data contributed to this strong performance.

- **SARIMAX and LSTM (Sequential):**  
  The SARIMAX model followed closely with an RMSE of 1231, benefiting from its capability to handle seasonality in the data. Meanwhile, the LSTM (Sequential) model also performed well, with an RMSE of 1178, though it slightly lagged behind the LSTM (Single) due to the challenges of training sequential layers effectively.

- **Moderate Performance of Prophet:**  
  Our Prophet model achieved an RMSE of 2482. While it captured seasonal trends effectively, it struggled with more complex, non-linear relationships within the data.

- **Least Effective Model:**  
  The ARIMA model showed the highest RMSE of 4579 among all tested models. Its limitations in handling seasonal and trend variations in larger datasets resulted in lower accuracy compared to other approaches.  

#### ***Insights:***  
- Electricity consumption exhibited relatively stable patterns, making it easier to predict.  
- Time-based features such as weekends and holidays significantly influenced electricity usage trends.  
- The use of lag features and weather data further enhanced the performance of advanced models like LSTM.  

#### **Gas Consumption:**  
| Model             | RMSE       |
|-------------------|------------|
| ARIMA             | 2936       |
| SARIMAX           | 2937       |
| LSTM Single       | 5266       |
| LSTM Sequential   | 2984       |
| Prophet           | 12197      |

**Gas Actual vs Fitted**  
![image](https://github.com/user-attachments/assets/bb349be9-59e9-4a10-924f-423e72ea5d1f)

#### ***Analysis:***  

- **Best-Performing Model:**  
  Our ARIMA model produced the lowest RMSE of 2936, showing that its simple, statistical approach was well-suited for gas consumption data, especially when dealing with short-term dependencies.

- **Similar Performance of SARIMAX:**  
  SARIMAX closely followed ARIMA with an RMSE of 2937. However, the inclusion of seasonal components didn’t offer significant improvements, possibly due to the inconsistent patterns in gas data.

- **LSTM Models:**  
  - The LSTM (Sequential) model performed moderately with an RMSE of 2984, slightly worse than ARIMA and SARIMAX. Its results suggest that while LSTMs can capture complex patterns, they struggled due to the high proportion of missing values and irregularities in gas usage.
  - The LSTM (Single) model had the poorest performance among all, with an RMSE of 5266. This indicates that the Single LSTM struggled to generalize well on this dataset.

- **Poor Performance of Prophet:**  
  Our Prophet model showed the highest RMSE of 12197, indicating that it struggled to handle the irregular and sparse nature of the gas consumption data.

#### ***Insights:***  
- Gas consumption predictions were more challenging due to missing data and irregular patterns.  
- Imputation of missing values (with 0) likely affected the models’ ability to accurately capture trends in gas usage.  
- Weather features, particularly temperature, had a significant influence on gas usage, with higher consumption during colder periods.  

#### ***Overall Observations:***  
- Electricity predictions were generally more accurate due to stable and consistent patterns in the data. Advanced models like LSTM Single and SARIMAX performed the best.  
- Gas predictions faced challenges due to irregularities and data gaps, with simpler models like ARIMA outperforming more complex methods.  

---

## 6. Running the Code (script)  
This project is implemented in a Jupyter Notebook. To run the code, simply follow these steps:

1. Open the Jupyter Notebook by running:

    ```bash
    jupyter notebook
    ```

2. Open the notebook file (`*.ipynb`) in the Jupyter interface.

3. Run the entire script by selecting **Run All** from the "Cell" menu.

The notebook will execute all the code, including model training and evaluation.

---

## 7. Team and Contact

- **Marij Qureshi**: MEng Aeronautical Engineering (Imperial), MSc Data Science (Brunel), ex-EY Parthenon
- **Georgios Gkakos**: MSc Data Science (Brunel), BSc Economics (AUTH)
- **Het Suhagiya**: MSc Data Science (Brunel), BSc Information Technology 

For questions, feel free to reach out via GitHub issues or email any of us.

---

## 8. License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
