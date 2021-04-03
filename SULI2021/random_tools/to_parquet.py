import pandas as pd
import os
import pyarrow

base = os.path.dirname(os.getcwd())
pq_dir = base + '/data/B3/parquet/'
csv_dir = input('Directory of text files: ')
files = os.listdir(csv_dir)

for file in files:
    df = pd.read_csv(os.path.join(csv_dir, file), delimiter=" ")
    df.columns = ['time', 'amplitude']
    print(f'{pq_dir}{file.split(".")[0]}.pq')
    df.to_parquet(f'{pq_dir}{file.split(".")[0]}.pq', engine='pyarrow')