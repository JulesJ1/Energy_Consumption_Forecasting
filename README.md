# Forecasting Energy Consumption Using Machine Learning

<img src="/assets/images/energyimage1.jpg">

## About This Project
The aim of this project is to explore how machine learning can be used to predict energy consumption as a tool for energy management. Energy forecasting is pivotal for assessing how different factors may affect energy production, consumption and demand. Energy consumption forecasts allow us to stay prepared for events in the future by managing the efficiency of power systems and requirements for different energy sources. According to the International Energy Agency, cost-effective energy efficiency improvements can have positive macroeconomic impacts, such as boosting economic activity which leads to increased employment. It can also have a positive impact on economic activity related to trade balances and energy prices.

Information about model performance and dataset analysis is available [here](https://github.com/JulesJ1/Energy_Generation/tree/main/scripts/README.md)

## Running The Code

#### Requirement: 

ENTSO-E API key - Running a clone requires an ENTSO-E API key to give access to the live energy consumption data. API keys are available by sending an email to transparency@entsoe.eu with “Restful API access” in the subject line.
The API key should then be stored in a file named ".env" including the following line:
```bash
ENTSOE_API_KEY = <YOUR API KEY>
```



Install the apps dependancies into your environment using:
```bash
pip install -r requirements.txt
```

Start the app by running the "dashapp.py" file, or through the terminal using:
```bash
python dashapp.py
```
## Dashboard

<img src="/assets/images/dashapp.JPG">


