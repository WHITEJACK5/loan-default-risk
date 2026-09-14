import pandas as pd
from pathlib import Path
for name in ['train','val','test']:
    p = Path(f'data/processed/{name}.parquet')
    df = pd.read_parquet(p)
    print(name, 'rows=', len(df),
          'bad_loan=', int((df['loan_amnt']<=0).sum()),
          'bad_dti=', int((df['dti']<0).sum()),
          'bad_target=', int((~df['target'].isin([0,1])).sum()),
          'null_year=', int(df['issue_year'].isna().sum()),
          'rate=', round(df['target'].mean(),3))
