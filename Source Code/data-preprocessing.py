import pandas as pd

dataset = pd.read_csv("tendulkarsp.csv")
#print(dataset)
del dataset['Start Date']
data_new = dataset.replace('-', "NaN")
data_new.to_csv("processed.csv", index=None)
data_processed = pd.read_csv("processed.csv")
data_cleaned = data_processed.dropna()
print(data_cleaned)
data_cleaned.to_csv("cleaned.csv", index=None)
print(type(dataset))
data = pd.read_csv("cleaned.csv")
