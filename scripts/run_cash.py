import src.policy.evaluate_profit as p
df=p.cash_table()
print(df.head(2).to_string())
print(f"approve-all {df['realized_profit'].sum():,.0f}")
