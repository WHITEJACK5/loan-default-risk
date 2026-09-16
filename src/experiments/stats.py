import scipy.stats as st
import numpy as np

def chi2_test(a_conv, a_n, b_conv, b_n):
    # 2x2 table [[conv, non-conv]]
    obs = np.array([[a_conv, a_n-a_conv],[b_conv, b_n-b_conv]])
    chi2, p, _, _ = st.chi2_contingency(obs)
    return {"chi2": chi2, "p": p, "sig": p<0.05}

def srm_check(n_a, n_b, exp_ratio=0.5):
    n=n_a+n_b
    exp_a, exp_b = n*exp_ratio, n*(1-exp_ratio)
    chi2 = (n_a-exp_a)**2/exp_a + (n_b-exp_b)**2/exp_b
    p = 1 - st.chi2.cdf(chi2,1)
    return {"chi2": chi2, "p": p, "srm": p<0.05}

if __name__=="__main__":
    print(chi2_test(120,1000,150,1000))
    print(srm_check(4963,5037))
    print(srm_check(5600,4400))
