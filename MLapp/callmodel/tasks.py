#from scripts import data
from callmodel.models import Totalload
from datetime import datetime, timedelta
import pytz
from celery import shared_task
import sys
from time import sleep

sys.path.append("C:/Users/Jules/OneDrive/Desktop/MLProjects/Energy_Generation")

from scripts import data

@shared_task
def my_task():
    for i in range(11):
        print(i)
        sleep(i)
    return "Task Complete"

@shared_task
def callAPI():
    
    tz = pytz.timezone('Europe/Madrid') 
    now = datetime.now(tz)

    currenthour = now.strftime("%Y-%m-%d %H:00:00")
  
    prevhour = now - timedelta(hours=12)
    prevhour = prevhour.strftime("%Y-%m-%d %H:00:00")
    

    Totalload.objects.all().delete()
    df = data.energy_api(prevhour,currenthour)
    for index,row in df.iterrows():
        Totalload.objects.create(date = index,totalload = row["Actual Load"])
    
    
    
    
    return "complete!"
 
