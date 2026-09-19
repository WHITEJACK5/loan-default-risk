import gradio as gr
import joblib
from pathlib import Path
import pandas as pd
MODEL = Path("artifacts/model.joblib")
def predict(loan_amnt, int_rate, annual_inc, dti, fico_low, revol_util):
    import numpy as np
    # if model exists use it, else heuristic (same as HF)
    if MODEL.exists():
        model = joblib.load(MODEL)
        df = pd.DataFrame([{"loan_amnt":loan_amnt,"int_rate":int_rate,"annual_inc":annual_inc,"dti":dti,"fico_range_low":fico_low,"fico_range_high":fico_low+4,"revol_util":revol_util,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","delinq_2yrs":0,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"total_acc":20,"term":"36 months","earliest_cr_line":"Jan-2005"}])
        p = float(model.predict_proba(df)[0,1])
    else:
        s = -4.5 + 0.12*int_rate + 0.04*dti + 0.015*revol_util -0.005*(fico_low-650)/10 -0.00001*annual_inc
        p = 1/(1+np.exp(-s))
    return f"PD {p:.3f} -> {'approve' if p < 0.05 else 'reject'} (thr 0.05)"

demo = gr.Interface(fn=predict, inputs=[gr.Number(value=10000, label="loan_amnt"), gr.Slider(6,30,value=13.5, label="int_rate"), gr.Number(value=65000, label="annual_inc"), gr.Slider(0,40,value=18, label="dti"), gr.Slider(600,850,value=680, label="fico_low"), gr.Slider(0,100,value=45, label="revol_util")], outputs="text", title="loan-default-risk")
demo.launch()