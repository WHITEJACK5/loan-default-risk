import hashlib

def assign(user_id: str, exp_id: str = "loan_policy_v1", salt: str = "v2-salt-42", allocation: float = 0.5) -> str:
    key = f"{salt}:{exp_id}:{user_id}".encode()
    h = int(hashlib.sha256(key).hexdigest(), 16) % 10000
    return "A" if h < allocation*10000 else "B"
