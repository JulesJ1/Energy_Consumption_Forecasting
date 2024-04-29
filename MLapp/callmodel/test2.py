from datetime import datetime, timedelta
import sys
import pytz



sys.path.append("C:/Users/Jules/OneDrive/Desktop/MLProjects/Energy_Generation")

from scripts import data

tz = pytz.timezone('Europe/Madrid') 
now = datetime.now(tz)

currenthour = now.strftime("%Y-%m-%d %H:00:00")

#print("prev",prevhour)
#df = data.energy_api(prevhour,currenthour)

prevhour = now - timedelta(hours=12)
prevhour = prevhour.strftime("%Y-%m-%d %H:00:00")
print("prev",prevhour)
print(currenthour)
df = data.energy_api(prevhour,currenthour)
#df = data.energy_api("2024-03-08 00:00:00","2024-03-08 01:00:00")
for index,row in df.iterrows():
    print(index)
    print(row["Actual Load"])
#print(df.index[0])
#print(df["Actual Load"].iloc[0])
