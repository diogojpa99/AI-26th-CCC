import pandas as pd

df = pd.read_csv('training_data.csv', names= ['dance','label'])

outliers = []
for i in range(len(df['dance'])):
    if len(df['dance'][i]) != 30:
        outliers.append(i)
        
        

