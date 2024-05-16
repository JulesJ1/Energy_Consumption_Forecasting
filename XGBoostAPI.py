import uvicorn
from fastapi import FastAPI
import gunicorn
import pickle
from Dates import Dates
import scripts.data as energy
import pandas as pd
from entsoe import EntsoePandasClient
import os


app = FastAPI()

with open('models/xgboost_v2_no_temp.joblib', 'rb') as pickle_file:
    model = pickle.load(pickle_file)


@app.post("/energydata")
def energydata(data:Dates):
    data = data.dict()
    starttime = data["starttime"]
    endtime = data["endtime"]

    starttime = pd.Timestamp(starttime, tz = 'Europe/Madrid' )
    endtime  = pd.Timestamp(endtime, tz = 'Europe/Madrid' )

    client = EntsoePandasClient(api_key= os.getenv("ENTSOE_API_KEY"))

    load  = client.query_load("ES", start = starttime, end=endtime)

    load = load.to_dict(orient="index")


    return load

@app.post("/predict")
def predict(data:Dates):
    pass

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port = 8000)

