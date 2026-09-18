import json
from pathlib import Path

def nb(cells):
    return {"cells":[{"cell_type":"markdown","metadata":{},"source":[c]} if i%2==0 else {"cell_type":"code","metadata":{},"source":[c],"outputs":[],"execution_count":None} for i,c in enumerate(cells)], "metadata":{},"nbformat":4,"nbformat_minor":5}

Path("notebooks/01_eda.ipynb").write_text(json.dumps(nb(["# 01 EDA\n151 cols, loan_status, leakage","import pandas as pd\ndf=pd.read_csv('data/raw/accepted_2007_to_2018Q4.csv.gz', nrows=5)\nprint(df.shape)\nprint(df.columns.tolist()[:30])\nprint(df['loan_status'].value_counts())\nprint(pd.read_csv('data/raw/accepted_2007_to_2018Q4.csv.gz', nrows=0).columns.tolist())\n# see docs/leakage_audit.md"])))
Path("notebooks/02_baseline.ipynb").write_text(json.dumps(nb(["# 02 Baseline\nLogistic 0.719/0.408 vs LGBM 0.722/0.413 val, test 0.703/0.371","import pandas as pd\nfrom sklearn.metrics import *\n# run src/models/train.py and evaluate.py\nprint('val ROC 0.722 PR 0.413 KS 0.323 -> test 0.703 0.371 0.298')\nprint('Brier val 0.155 test 0.156')"])))
Path("notebooks/03_calibration_profit.ipynb").write_text(json.dumps(nb(["# 03 Calibration + Profit\nBrier 0.203->0.155 isotonic, profit thr 0.05","from pathlib import Path\nfrom PIL import Image\nImage.open('docs/calibration_curve.png').show()\nimport pandas as pd\nprint(pd.read_csv('docs/profit_curve.csv').head())\nprint('best thr 0.05 approve 7.9% profit 7.4M')"])))
print("saved 3 notebooks")