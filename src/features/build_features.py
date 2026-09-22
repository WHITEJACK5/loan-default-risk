from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

NUM_A = ["loan_amnt","int_rate","installment","annual_inc","dti","fico_range_low","fico_range_high","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs"]
NUM_B = ["loan_amnt","annual_inc","dti","fico_range_low","fico_range_high","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs"]
CAT_A = ["term","grade","sub_grade","emp_length","home_ownership","verification_status","purpose","addr_state"]
CAT_B = ["term","emp_length","home_ownership","verification_status","purpose","addr_state"]

def get_preprocessor(for_trees: bool = True, lender_side: str = "A"):
    NUM = NUM_B if lender_side=="B" else NUM_A
    CAT = CAT_B if lender_side=="B" else CAT_A
    if for_trees:
        num = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    else:
        from sklearn.preprocessing import StandardScaler
        num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)], remainder="drop")
