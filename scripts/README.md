# Exploratory Data Analysis

## Datasets
### Consumption Data
ENTSO-E, the European Network of Transmission System Operators for Electricity, provides a free API which gives users access to historical and current data from different TSOs around Europe. The power consumption data used in this project is taken from the ENTSO-E API, and consists of the hourly total consumption data for Spain. 
<img src="../images/dataset.png">
### Weather Data
The weather dataset is taken from the OpenWeather API and contains hourly weather data including metrics such as temperature, wind speed and cloud coverage for the corresponding consumption data timestamps in Madrid, Spain.
## Time Series

### Lag Features

### Seasonality


## XGBoost Algorithm Results

| First Header  |      MAE      |
| ------------- | ------------- |
| XGBoost + Lags only  | 1242      |
| XGBoost + Lags + Tuning Paramters  | 1242    |
| XGBoost + Lags + Tuning Paramters + Exo Variables (excluding weather data)  | 1242    |
| XGBoost + Lags + Tuning Paramters + Exo Variables (including weather data)   | 1242    |

## Next Steps
To further the development of this project, the goals for continuing the development of this project include:
* Creating a flask API that will be used to serve the models to the web app.
* Deployment of an additional model that will make predictions for daily load data.
* Testing and deployment of an LTSM model for the data using Pytorch and AWS.
  
