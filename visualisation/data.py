import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../energy_dataset.csv")
print(df.columns)
print(df.head)
columns = ["generation hydro pumped storage aggregated", "forecast wind offshore eday ahead", "generation fossil coal-derived gas", "generation wind offshore", "generation marine", "generation geothermal",
"generation fossil peat","generation fossil oil shale"]
df = df.drop(columns,axis = 1)
#plt.matshow(df.corr())
#df.fillna(df.mean(),inplace=True)
df.fillna(df.interpolate(method="linear"),inplace=True)
#plt.show()
#print(df.mean())

#print(df.var())
df["time"] = pd.to_datetime(df["time"],format = "%Y-%m-%d %H:%M:%S")
df['time'] = df['time'].apply(lambda x: x.replace(tzinfo=None))
df["time"] = pd.to_datetime(df["time"],format="ISO8601")

df = df.set_index('time')

plt.plot(df.index, df['generation biomass'])

plt.show()
