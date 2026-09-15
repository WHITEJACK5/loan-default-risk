from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler

NUM = ["loan_amnt","int_rate","installment","annual_inc","dti","fico_range_low","fico_range_high","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs"]
CAT = ["term","grade","sub_grade","emp_length","home_ownership","verification_status","purpose","addr_state"]

def get_preprocessor():
    num = Pipeline([("impute" , SimpleImputer(strategy="median")) , ("scale" , StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num" , num , NUM) , ("cat" , cat , CAT)],remainder = "drop")

