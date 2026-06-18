import pandas as pd

df = pd.read_excel('usstates50.xlsx')
print(list(df.columns))
print(df.loc[df['state'].astype(str).str.lower() == 'california'].head(1).to_dict('records'))
