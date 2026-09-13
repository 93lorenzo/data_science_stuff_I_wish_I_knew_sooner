# Data Science stuff I wish I knew sooner: Fraud Date Delta

For a long time I treated a fraud label the same way I treated any other column: it's either `0` or `1`,
end of story. What I was missing is that a fraud label is not a property of the transaction alone, it is
a property of the transaction **and of how long you waited before trusting the label**. Fraud is
practically never known instantly, it is *reported*, days or weeks after the transaction happened, by a
customer disputing a charge, a chargeback landing, or a fraud team closing a case. Until that report
arrives, a fraud that already happened still looks, in your data, exactly like a legitimate transaction.

That gap between "the transaction happened" and "we found out it was fraud" is the **date delta** this
article is about. Get it wrong and one of two things happens: cut the observation window too short and
you train on transactions the model calls "legitimate" simply because fraud hadn't been reported yet
(pure label noise, and it concentrates exactly on the most recent data, the part closest to production).
Cut it too long and you throw away months of otherwise usable history waiting for stragglers that were
never going to show up.

The companion notebook
[`5-Fraud-Date-Delta-Training.ipynb`](5-Fraud-Date-Delta-Training.ipynb) walks through measuring that lag,
finding a defensible observation window from it, and the two ways to turn that window into a label, only
one of which doesn't quietly poison the training set. On purpose, **no model is trained** in this notebook,
this is a labelling-strategy article, and the plots and counts are the whole point.

---

## Dataset

[Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
(Kaggle) — 2,512 bank transactions spanning 2023-01-02 to 2024-01-01, with amounts, channels, devices,
locations and a `TransactionDate`. It ships with **no fraud label**, which turns out to be convenient
here: every property of the label discussed below was put there on purpose, nothing is hiding inside a
pre-existing column.

Two fields are added synthetically, both fully reproducible from a single seeded
`numpy.random.Generator`:

* **`IsFraud`** — exactly 1% of rows, chosen with `rng.choice(..., replace=False)`. Using `choice` on row
  positions pins the *count* down exactly, rather than leaving it as an expectation that a coin-flip
  approach (`rng.random() < 0.01`) would only hit approximately.
* **`FraudReportedDate`** — for fraud rows only, `TransactionDate + lag`, where `lag` (in days) is drawn
  from a shifted exponential distribution: most fraud is spotted quickly, a long tail is spotted much
  later. This lag is the whole article, everything below is about how to handle it correctly.

```python
fraud_positions = rng.choice(n_rows, size=n_fraud, replace=False)
lag_days = np.clip(np.round(1 + rng.exponential(scale=50.0, size=n_fraud)), 1, 180)
fraud_date = transaction_date[fraud_positions] + pd.to_timedelta(lag_days, unit="D")
```

---

## 1. How long does it take to spot fraud?

![Distribution of the fraud reporting lag](img/delta_distribution.png)

Plotting `FraudReportedDate - TransactionDate` for the 25 fraud cases shows the shape the generator was
designed to produce: a cluster of quick detections in the first few weeks, a median lag of 26 days, and a
thinning tail stretching out past four months. Real fraud-reporting lags look like this too — most
disputes and chargebacks land quickly, but a minority take much longer to surface.

A single "days since transaction" cutoff, picked without looking at this distribution, will either sit
inside the bulk of the tail and silently miss it, or sit far out in the tail and throw away most of the
dataset waiting for stragglers that make up a small fraction of cases.

---

## 2. Finding the optimal date delta

Order transactions by date and call `reference_date` the last transaction date in the data (the end of
the training window). For a candidate window of `d` days:

* a transaction only counts as **mature** for `d` if it happened at least `d` days before
  `reference_date`. Anything more recent hasn't had the full `d` days to be reported yet, so nothing can
  honestly be said about it at that window length;
* among mature transactions, a fraud is **captured** at `d` if it was reported within `d` days of the
  transaction.

At `d = 1`, only frauds reported the next day, among transactions old enough to have had that one day to
prove it, count. At `d = 2`, it's frauds reported within 2 days, among transactions that are at least 2
days older than the end of training, and so on. Growing `d` cuts both ways: it lets in fraud with a
longer reporting lag, but it also shrinks the pool of mature transactions, since the most recent `d` days
keep getting excluded every time `d` grows.

```python
for d in range(1, max_days + 1):
    cutoff = reference_date - pd.Timedelta(days=d)
    is_mature = transaction_date <= cutoff
    is_captured = is_mature & (label == 1) & (date_delta_days <= d)
    captured_frauds[d] = is_captured.sum()

optimal_date_delta = captured_frauds.idxmax()
```

![Matured frauds captured against the candidate observation window](img/maturity_curve.png)

That tension is not incidental, it is what gives this curve its shape in general: **it goes up, then
down**. It climbs steeply while short windows are still missing most of the reporting-lag distribution,
flattens into a plateau once nearly every reportable fraud in the mature population has been captured,
then drifts back down as growing `d` keeps shrinking the mature pool for no further gain, occasionally
ticking back up on the strength of one more straggler before falling again. **60 days** is the optimal
date delta here, the smallest window that captures as many mature frauds (17) as any window realistically
will, this dataset's curve even touches that same value a second time around day 79, at the cost of over
150 extra dropped transactions for nothing extra in return.

That "smallest window" qualifier matters. Several candidate values of `d` typically tie for the peak;
`idxmax` always returns the first one it finds, which is also the cheapest: a larger, tied `d` captures
exactly the same fraud while discarding strictly more mature transactions. Ties should always break
toward the earliest `d`, never arbitrarily.

### The last label is rarely the cheapest one

Reaching the global peak isn't free even before ties enter the picture. Tracking the earliest `d` that
reaches each new best captured-fraud count, and what it cost in dropped transactions, tells a diminishing-returns
story:

| Captured frauds | Earliest `d` (days) | Mature transactions | Dropped |
|---|---|---|---|
| 12 | 26 | 2,342 | 170 |
| 13 | 29 | 2,305 | 207 |
| 14 | 30 | 2,305 | 207 |
| 15 | 31 | 2,304 | 208 |
| 16 | 33 | 2,287 | 225 |
| **17** | **60** | **2,085** | **427** |

Every step from 12 to 16 captured frauds costs a handful of additional dropped rows. The very last step,
16 to 17, costs **202 more rows on its own — almost as much as every earlier step combined** — for exactly
one extra fraud case. `get_optimal_date_delta` still returns the strict global optimum (60 days) by
default, and that is the right default to ship with. But a team looking at this table is not wrong to
decide the last row's price tag isn't worth it, and to pick `d = 33` instead: 16 captured frauds for 225
dropped rows, rather than 17 for 427. Either choice is defensible; the mistake would be picking `d`
without ever seeing this trade-off.

---

## 3. Turning a date delta into a label — two ways, only one is safe

Picking `optimal_date_delta = 60` answers "how many days of reporting lag should the label definition
tolerate?". Applying that answer to the training data can be done two ways, and this is the part worth
being deliberate about:

1. **Discard the immature tail (recommended).** Drop every transaction from the last 60 days of the
   dataset outright. We genuinely do not know yet whether they will turn into fraud within the window, so
   they are excluded rather than guessed at. Everything that remains has had the full window to be
   reported, so its label — fraud or not — can be trusted.
2. **Relabel and keep everything.** Keep the immature tail too, and mark anything not (yet) reported as
   fraud as legitimate (`0`).

Both behaviours also relabel any fraud reported *later* than the window as `0` — that part is just
enforcing the window's own definition of "fraud" consistently, in both cases equally. The two only differ
on transactions too recent to have finished the observation window.

| | Rows | Fraud count | Fraud ratio |
|---|---|---|---|
| Original (synthetic ground truth) | 2,512 | 25 | 0.995% |
| **Discard immature tail** | 2,085 (427 dropped) | 17 | 0.815% |
| **Relabel and keep everything** | 2,512 (0 dropped) | 21 | 0.836% |

The 427 rows in the immature tail split two ways: 4 of them are transactions already confirmed as fraud,
just too recently for the 60-day window to have fully elapsed, and 421 are transactions that simply
haven't had 60 days to be reported yet. Neither mode is free of cost, but the costs are different in
kind:

* **Discarding** loses those 4 already-confirmed frauds along with the rest of the tail — a real, but
  small and fully known cost. Every row that remains earned its label honestly.
* **Relabelling** recovers those same 4 frauds (fraud count goes from 17 back up to 21), but it also
  keeps the other 421 rows and labels every one of them legitimate by default. None of them happen to
  have flipped to fraud in this particular snapshot, but the mode never checked; it would treat them the
  exact same way whether they had or not. That is the actual risk: nothing about "relabel and keep
  everything" distinguishes "confirmed clean" from "not yet resolved", for these 421 rows or for any
  future slice of data where the distinction might not resolve as harmlessly.

```python
strict_df  = analyzer.apply_date_delta_window(optimal_date_delta, discard_immature=True)
relabel_df = analyzer.apply_date_delta_window(optimal_date_delta, discard_immature=False)
```

---

## What to do about it

A fraud definition is a **time-bound** definition, "fraud, if reported within `d` days", whether a team
writes that down explicitly or not. Once `d` is chosen:

1. **Measure the reporting lag before picking a window.** A histogram of the lag distribution
   (Section 1) tells you whether a given `d` is realistic; picking one blind is picking blind twice, once
   for the cutoff and once for how much of the tail it actually covers.
2. **Never default an unproven row to "not fraud".** The temptation is to keep every row because it
   feels like free data. It isn't. It's label noise, and it concentrates on the rows closest to
   production, exactly where a model's mistakes are most expensive.
3. **Discarding the immature tail is a conservative choice, not a lossy one.** It costs recent rows,
   sometimes even a handful of already-confirmed frauds sitting right at the edge of the window. That
   cost is small and known. The cost of mislabelled rows is neither.
4. **Expect the maturity curve to peak, and break ties toward the smallest `d`.** It climbs, then falls,
   because the two effects it balances (more lag tolerated, less mature data left) pull in opposite
   directions. When several windows tie for the best fraud count, the smallest one is strictly better,
   never just a tie-breaking convention.
5. **Look at the price of the last label before paying it.** The step from the second-best window to the
   true optimum is often far more expensive, in dropped transactions, than every earlier step combined.
   Taking the strict optimum by default is reasonable; taking it without checking what it cost is not.
6. **Revisit `d` as reporting behaviour changes.** A fraud team that gets faster (or slower) at closing
   cases shifts the whole lag distribution, and the optimal date delta with it. This isn't a constant to
   set once.

---

## Running the notebook

```bash
cd 5-fraud-date-delta-training
uv sync
uv run jupyter notebook
```

Or in VS Code: open `5-Fraud-Date-Delta-Training.ipynb` and select the `.venv` kernel.

---

## References

- [Bank Transaction Dataset for Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
- [numpy.random.Generator](https://numpy.org/doc/stable/reference/random/generator.html)
