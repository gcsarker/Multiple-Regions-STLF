import pandas as pd
import os

#Loading Load Demand Dataset
root_dir = '..'
demand_data_path = 'dataset/raw_demand_data.csv'
df = pd.read_csv(os.path.join(root_dir, demand_data_path))
print(df.shape)

