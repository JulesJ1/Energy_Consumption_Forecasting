import requests
import json
import pandas as pd
from io import StringIO  

params = {
    "starttime":"2024-05-15 12:00:00",
    "endtime":"2024-05-15 17:00:00",
    "steps":0
}
#data = requests.post("http://127.0.0.1:8000/energydata",params=params)
data = requests.post("http://127.0.0.1:8000/energydata",json=params)

data = pd.read_json(StringIO(data.content),orient="index")
print(data)