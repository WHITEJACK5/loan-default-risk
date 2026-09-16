import hashlib

def assign(user_id: str, exp_id: str = "loan_policy_v1") -> str:
    key = f"{user_id}:{exp_id}".encode()
    h = int(hashlib.md5(key).hexdigest(), 16) % 100
    return "A" if h < 50 else "B"

def check_srm(assignments):
    from collections import Counter
    c = Counter(assignments)
    n = len(assignments)
    # chi2 vs 50/50, df=1, p<0.05 -> SRM
    import scipy.stats as st
    obs = [c.get("A",0), c.get("B",0)]
    exp = [n*0.5, n*0.5]
    chi2 = sum((o-e)**2/e for o,e in zip(obs, exp) if e>0)
    p = 1 - st.chi2.cdf(chi2, 1)
    return {"A": obs[0], "B": obs[1], "chi2": chi2, "p": p, "srm": p<0.05}

if __name__=="__main__":
    ids = [f"user_{i}" for i in range(10000)]
    assigns = [assign(uid) for uid in ids]
    print(check_srm(assigns))
    # sticky check
    print(assign("user_42"), assign("user_42"), assign("user_42")==assign("user_42"))
