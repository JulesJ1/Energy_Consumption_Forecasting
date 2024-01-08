import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../energy_dataset.csv")
print(df.columns)
print(df.head)
columns = ["generation hydro pumped storage aggregated", "forecast wind offshore eday ahead", "generation fossil coal-derived gas", "generation wind offshore", "generation marine", "generation geothermal",
"generation fossil peat","generation fossil oil shale"]
df = df.drop(columns,axis = 1)
#plt.matshow(df.corr())
df.fillna(df.mean(),inplace=True)
#plt.show()
print(df.mean())

print(df.var())