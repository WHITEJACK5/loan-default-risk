import pandas as pd
from pathlib import Path
df = pd.read_csv("data/raw/accepted_2007_to_2018Q4.csv.gz", usecols=["issue_d","term","loan_status"], low_memory=False)
df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
df["term_m"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
rows=[]
for (y,t), g in df.groupby(["issue_year","term_m"]):
    total=len(g)
    resolved=g[g["loan_status"].isin(["Fully Paid","Charged Off"])]
    dropped=total-len(resolved)
    dr=resolved["loan_status"].eq("Charged Off").mean() if len(resolved)>0 else 0
    rows.append(f"| {y} | {t}m | {total} | {len(resolved)} | {dropped} | {dr:.3f} |")
Path("docs/censoring.md").write_text("# Censoring audit\n| issue_year | term | total | resolved | dropped | dr_resolved |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
print("saved docs/censoring.md", len(rows))
