import pandas as pd


dataset_path = "2017-SUEE-data-set/data.csv"
df = pd.read_csv(dataset_path)

print(df.head())  
