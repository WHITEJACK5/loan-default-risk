# Leakage Audit - Loan Default Risk

## Rule
If known only AFTER funding/payment starts -> DROP. Only origination-time stays.

## Dropped - identifiers
| col | reason |
|-----|--------|
| id, member_id, url, desc, title, zip_code, emp_title | IDs / free text, not scoring signals |
| pymnt_plan, policy_code | constant / post-plan |

## Dropped - post-funding money (target leak)
| col | reason |
|-----|--------|
| out_prncp, out_prncp_inv | remaining principal after payments |
| total_pymnt, total_pymnt_inv | total paid so far |
| total_rec_prncp, total_rec_int, total_rec_late_fee | received breakdown post-funding |
| recoveries, collection_recovery_fee | collections after default |
| last_pymnt_d, last_pymnt_amnt, next_pymnt_d | payment history |
| last_credit_pull_d, last_fico_range_high, last_fico_range_low | credit pull after funding |
| hardship_flag, hardship_type, hardship_reason, hardship_status, deferral_term, hardship_amount, hardship_start_date, hardship_end_date, payment_plan_start_date, hardship_length, hardship_dpd, hardship_loan_status, orig_projected_additional_accrued_interest, hardship_payoff_balance_amount, hardship_last_payment_amount | hardship program post-funding |
| debt_settlement_flag, debt_settlement_flag_date, settlement_status, settlement_date, settlement_amount, settlement_percentage, settlement_term | settlement post-default |
| disbursement_method | keep? cash vs direct - origination, KEEP actually - remove from drop |

## Kept - origination examples
loan_amnt, term, int_rate, installment, grade, sub_grade, emp_length, home_ownership, annual_inc, verification_status, issue_d, loan_status, purpose, addr_state, dti, delinq_2yrs, earliest_cr_line, fico_range_low, fico_range_high, inq_last_6mths, open_acc, revol_util, total_acc
