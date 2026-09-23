import scipy.stats as st
import numpy as np
import math

def chi2_test(a_conv, a_n, b_conv, b_n):
    obs = np.array([[a_conv, a_n-a_conv],[b_conv, b_n-b_conv]])
    chi2, p, _, _ = st.chi2_contingency(obs)
    return {"chi2": chi2, "p": p, "sig": p<0.05}

def srm_check(n_a, n_b, exp_ratio=0.5):
    n=n_a+n_b
    exp_a, exp_b = n*exp_ratio, n*(1-exp_ratio)
    chi2 = (n_a-exp_a)**2/exp_a + (n_b-exp_b)**2/exp_b
    p = 1 - st.chi2.cdf(chi2,1)
    # alpha 0.001 per Work Plan
    return {"chi2": chi2, "p": p, "srm": p<0.001}

def z_test(a_conv, a_n, b_conv, b_n):
    p1, p2 = a_conv/a_n, b_conv/b_n
    p = (a_conv+b_conv)/(a_n+b_n)
    se = math.sqrt(p*(1-p)*(1/a_n+1/b_n))
    z = (p1-p2)/se if se>0 else 0
    pval = 2*(1-st.norm.cdf(abs(z)))
    return {"z": z, "p": pval}

def sample_size(p1, p2, alpha=0.05, beta=0.2):
    z_a = st.norm.ppf(1-alpha/2)
    z_b = st.norm.ppf(1-beta)
    return ((z_a+z_b)**2 * (p1*(1-p1)+p2*(1-p2)) / (p1-p2)**2) if p1!=p2 else 0

if __name__=="__main__":
    print(chi2_test(120,1000,150,1000))
    print(srm_check(4963,5037))
    print(z_test(120,1000,150,1000))
    print(sample_size(0.12,0.15))
