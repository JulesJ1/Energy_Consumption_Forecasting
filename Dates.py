
from pydantic import BaseModel,ConfigDict
from datetime import datetime
import pandas as pd

class Dates(BaseModel):
    starttime: datetime = None
    endtime:   datetime = None
    


class Predictions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Lastwindow: pd.DataFrame
    steps:  int