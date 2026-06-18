import pandas as pd
from pathlib import Path

df = pd.read_excel('usstates50.xlsx')
row = df.loc[df['state'].astype(str).str.lower() == 'california'].iloc[0]
print(row[['state', 'l_source']].to_dict())
