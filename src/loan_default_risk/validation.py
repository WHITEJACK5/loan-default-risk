import pandera as pa
import pandas as pd

schema = pa.DataFrameSchema({
    "loan_amnt": pa.Column(int, pa.Check.gt(0)),
    "dti": pa.Column(float, pa.Check.ge(0)),
    "fico_range_low": pa.Column(int, pa.Check.between(300,850)),
    "int_rate": pa.Column(float, pa.Check.between(5,35)),
    "annual_inc": pa.Column(int, pa.Check.gt(0)),
})

def validate(df: pd.DataFrame) -> pd.DataFrame:
    return schema.validate(df, lazy=True)