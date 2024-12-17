# ML-Project

# 1. Intoduction
Why and what are we doing in this project? narrative of the project. Intorduce the team work and what was the tasks for all the team member

# 2. Key Features
model used, teck stack used, dataset used

# 3. Data Prep
data cleaning, feature engineering, table of features used. 

# 4. ML Models
brief definition about all the models used and why is it used.

# 5. Results

- **ARIMA Model**:
The ARIMA model did a decent job at predicting both electricity and gas usage. However, it struggled more with gas data because it's a simpler model that can miss complex patterns.

- **SARIMAX Model**:
SARIMAX was excellent at predicting electricity usage since it can use additional variables to improve accuracy. However, it didn’t work well for gas predictions, likely because gas patterns are more complex.

- **LSTM Single Layer**:
The single-layer LSTM didn’t perform well for either electricity or gas predictions. It likely overfit the training data, meaning it learned too much from the training set and couldn’t handle new data effectively.

- **LSTM Sequential**:
The sequential LSTM did great for electricity predictions because it’s good at learning patterns over time. However, it struggled with gas data, likely because gas usage patterns are less predictable and harder to model.

- 6. Running the code (script)
explain how to run the code, dependencies and requirements

- 7. Furture Work
what else can be done, new methods (if any), potential improvements in the current work.

- 8. Licence
open source. anyone can contrbute
