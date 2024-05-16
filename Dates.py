
from pydantic import BaseModel
from datetime import datetime
from entsoe import EntsoePandasClient

class Dates(BaseModel):
    starttime: datetime = None
    endtime:   datetime = None
    steps:     int