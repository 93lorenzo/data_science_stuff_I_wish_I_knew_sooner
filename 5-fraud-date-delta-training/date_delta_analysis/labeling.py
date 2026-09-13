from __future__ import annotations

import numpy as np
import pandas as pd


def add_synthetic_fraud_fields(
    df: pd.DataFrame,
    transaction_date_col: str = "TransactionDate",
    label_col: str = "IsFraud",
    fraud_date_col: str = "FraudReportedDate",
    fraud_rate: float = 0.01,
    min_lag_days: int = 1,
    max_lag_days: int = 60,
    lag_scale: float = 8.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Attach a synthetic fraud label and a synthetic fraud-reported date.

    The dataset this repo pulls from has no fraud label at all, so two fields
    are fabricated for the sake of the article:

    1. ``label_col`` — exactly ``fraud_rate`` of the rows are marked as fraud,
       chosen with ``rng.choice`` (no replacement) so the ratio is exact
       rather than an expected value.
    2. ``fraud_date_col`` — for fraud rows only, ``transaction_date + lag``,
       where ``lag`` is drawn from a shifted exponential (most fraud is
       spotted quickly, a long tail is spotted much later). This lag is the
       whole point of the article: it is the gap the "date delta" logic has
       to reason about.

    Both draws come from the same seeded ``numpy.random.Generator``, so the
    same ``seed`` always reproduces the same label assignment and the same
    lags.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    n_fraud = int(round(n_rows * fraud_rate))

    fraud_positions = rng.choice(n_rows, size=n_fraud, replace=False)
    label = np.zeros(n_rows, dtype=int)
    label[fraud_positions] = 1

    lag_days = np.round(min_lag_days + rng.exponential(scale=lag_scale, size=n_fraud))
    lag_days = np.clip(lag_days, min_lag_days, max_lag_days).astype(int)

    fraud_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    transaction_dates = df[transaction_date_col].to_numpy()[fraud_positions]
    fraud_date.iloc[fraud_positions] = transaction_dates + pd.to_timedelta(lag_days, unit="D")

    out = df.copy()
    out[label_col] = label
    out[fraud_date_col] = fraud_date
    return out
