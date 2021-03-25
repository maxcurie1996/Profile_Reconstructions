import pandas as pd
import os
import pyarrow


base = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet'
files = os.listdir(base)

for file in files:
    df = pd.read_csv(os.path.join(base, file), delimiter=" ")
    df.columns = ['time', 'amplitude']
    print(f'/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/{file.split(".")[0]}.pq')
    df.to_parquet(f'/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/{file.split(".")[0]}.pq', engine='pyarrow')