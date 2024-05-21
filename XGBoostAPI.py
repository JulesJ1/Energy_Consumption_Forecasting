import uvicorn
from fastapi import FastAPI
import gunicorn
import pickle
from Dates import Dates,Predictions
import scripts.data as energy
import pandas as pd
from entsoe import EntsoePandasClient
import os
from datetime import datetime, timedelta

app = FastAPI()



@app.on_event('startup')
def load_model():
    with open('models/xgboost_v2_no_temp.joblib', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

#Pulls 
@app.post("/energydata")
def energydata(data:Dates):
    data = data.dict()
    starttime = data["starttime"]
    endtime = data["endtime"]

    starttime = pd.Timestamp(starttime, tz = 'Europe/Madrid' )
    endtime  = pd.Timestamp(endtime, tz = 'Europe/Madrid' )

    client = EntsoePandasClient(api_key= os.getenv("ENTSOE_API_KEY"))

    load  = client.query_load("ES", start = starttime, end=endtime)

    load.index = pd.to_datetime(load.index)
    load = load[load.index.minute == 00]
    load = load.to_dict()


    return load

@app.post("/predict")
def predict(data:Predictions):
    lags =  data.Lastwindow

    now = datetime.now()

    lags = pd.DataFrame.from_dict(lags)
   
    lags.index = pd.to_datetime(lags.index)
 
    if lags.index.tzinfo != None:
        lags.index.tz_convert(tz = "utc")
    lags.index = lags.index.tz_localize(None)
    lags = lags.asfreq("1h")

    steps = data.steps
    

    with open('models/xgboost_v2_no_temp.joblib', 'rb') as pickle_file:
        model1 = pickle.load(pickle_file)

    
    start = now.strftime("%Y-%m-%d %H:00:00")
    end = now + timedelta(hours=steps)
    end = end.strftime("%Y-%m-%d %H:00:00")
    i = pd.date_range(start,end,freq = "1h")

    exo_df = pd.DataFrame(index = i)

    poly_cols =     ['sin_month_1', 
            'cos_month_1',
            'sin_week_of_year_1',
            'cos_week_of_year_1',
            'sin_week_day_1',
            'cos_week_day_1',
            'sin_hour_day_1',
            'cos_hour_day_1',
            'daylight_hours',
            'is_daylight']

    exog = energy.create_features(exo_df,poly_cols)

    features = []

    features.extend(exog.filter(regex='^sin_|^cos_').columns.tolist())

    features.extend(exog.filter(regex='^temp_.*').columns.tolist())

    features.extend(['temp'])
    exog = exog.filter(features, axis=1)
 
    features = [x for x in features if "temp" not in x]

    prediction = model1.predict(
                steps = steps,
                last_window = lags["Actual Load"],
                exog = exog[features]
            )

    
    prediction.columns = ["Actual Load"]
    

    return prediction.to_dict()
    

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port = 8000)

