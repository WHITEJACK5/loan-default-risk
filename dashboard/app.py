import streamlit as st
import pandas as pd
st.title('loan-default-risk dashboard')
st.write('profit curve')
st.line_chart(pd.read_csv('docs/profit_curve.csv'))

