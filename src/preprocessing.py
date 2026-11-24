
import numpy as np


def feature_engineering(data):

    data = data.copy()

    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]

    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

    pay_status_cols = [
        'PAY_0', 'PAY_2', 'PAY_3',
        'PAY_4', 'PAY_5', 'PAY_6'
    ]

    data['AVG_BILL_AMT'] = data[bill_cols].mean(axis=1)

    data['TOTAL_BILL_AMT'] = data[bill_cols].sum(axis=1)

    data['AVG_PAY_AMT'] = data[pay_amt_cols].mean(axis=1)

    data['TOTAL_PAY_AMT'] = data[pay_amt_cols].sum(axis=1)

    data['CREDIT_UTILIZATION'] = (
        data['AVG_BILL_AMT'] /
        data['LIMIT_BAL'].replace(0, np.nan)
    ).fillna(0)

    data['PAYMENT_BILL_RATIO'] = (
        data['TOTAL_PAY_AMT'] /
        data['TOTAL_BILL_AMT'].abs().replace(0, np.nan)
    ).fillna(0)

    data['AVG_REPAY_DELAY'] = data[pay_status_cols].mean(axis=1)

    data['MAX_REPAY_DELAY'] = data[pay_status_cols].max(axis=1)

    data['DELAYED_MONTHS'] = (
        data[pay_status_cols] > 0
    ).sum(axis=1)

    return data
