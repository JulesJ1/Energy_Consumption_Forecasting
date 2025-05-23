
from pydantic import BaseModel,ConfigDict, Json
from datetime import datetime
from typing import Any

class Dates(BaseModel):
    starttime: datetime = None
    endtime:   datetime = None
    


class Predictions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Lastwindow: Json[Any]
    steps:  int