from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FraudDateDeltaAnalyzer:
    """Plot and reason about the lag between a transaction and its fraud report.

    Every fraud label used for training is a snapshot in time: "as far as we
    know today, this transaction is/isn't fraud". A transaction that happened
    yesterday hasn't had the same chance to be reported as fraud as one that
    happened six months ago. This class quantifies that lag (the "date
    delta") and helps pick an observation window that is long enough to
    trust the label, without discarding more history than necessary.
    """

    PALETTE = {
        "primary": "#5b8dd9",
        "positive": "#e05c5c",
        "neutral": "#888888",
        "highlight": "#f5a623",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        transaction_date_col: str = "TransactionDate",
        fraud_date_col: str = "FraudReportedDate",
        label_col: str = "IsFraud",
    ):
        self.transaction_date_col = transaction_date_col
        self.fraud_date_col = fraud_date_col
        self.label_col = label_col

        self.reference_date = df[transaction_date_col].max()
        self.df = df.assign(
            _date_delta_days=(df[fraud_date_col] - df[transaction_date_col]).dt.days
        )

    @property
    def fraud_deltas(self) -> pd.Series:
        """Date delta (days), one row per known fraud case."""
        is_fraud = self.df[self.label_col] == 1
        return self.df.loc[is_fraud, "_date_delta_days"]

    # ------------------------------------------------------------------
    # 1. The raw lag distribution
    # ------------------------------------------------------------------

    def plot_delta_distribution(self, bins: int = 20, title: str | None = None) -> plt.Figure:
        """Histogram of how many days after the transaction fraud was spotted."""
        deltas = self.fraud_deltas

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(deltas, bins=bins, color=self.PALETTE["positive"], edgecolor="white")
        ax.axvline(
            deltas.median(),
            color=self.PALETTE["neutral"],
            ls="--",
            lw=1.5,
            label=f"Median lag = {deltas.median():.0f} days",
        )
        ax.set_xlabel("Days between the transaction and its fraud report")
        ax.set_ylabel("Number of fraud cases")
        ax.set_title(title or "How long does it take to spot fraud?", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. The maturity curve and the optimal date delta
    # ------------------------------------------------------------------

    def compute_maturity_curve(self, max_days: int | None = None) -> pd.DataFrame:
        """Matured frauds captured at every candidate date delta.

        For a candidate window of ``d`` days, a transaction only counts if it
        is "mature" for that window, i.e. it happened at least ``d`` days
        before ``reference_date`` (the last transaction date in the data) —
        otherwise there simply hasn't been enough time yet to know whether it
        will turn into fraud within ``d`` days. Among mature transactions,
        a fraud "counts" for window ``d`` only if it was reported within
        ``d`` days of the transaction.

        Growing ``d`` cuts both ways: it lets in fraud with a longer lag, but
        it also shrinks the pool of mature transactions (the most recent
        ``d`` days are dropped every time), which can push some already
        mature frauds back out of the window. That tension is what produces
        a peak instead of a monotonically increasing curve.
        """
        deltas = self.fraud_deltas
        if max_days is None:
            max_days = int(deltas.max())

        rows = []
        for d in range(1, max_days + 1):
            cutoff = self.reference_date - pd.Timedelta(days=d)
            is_mature = self.df[self.transaction_date_col] <= cutoff
            is_captured = (
                is_mature
                & (self.df[self.label_col] == 1)
                & (self.df["_date_delta_days"] <= d)
            )
            n_mature = int(is_mature.sum())
            n_captured = int(is_captured.sum())
            rows.append(
                {
                    "date_delta_days": d,
                    "mature_transactions": n_mature,
                    "captured_frauds": n_captured,
                    "captured_fraud_rate": n_captured / n_mature if n_mature else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def get_optimal_date_delta(self, max_days: int | None = None) -> tuple[int, pd.DataFrame]:
        """Smallest date delta that captures the highest number of matured frauds.

        The maturity curve is rarely a clean climb, more fraud captured
        almost always means a stricter maturity cutoff too, so it typically
        rises, peaks, and drifts back down as ``d`` grows further. Several
        values of ``d`` can tie for that peak; ``idxmax`` always returns the
        first one, i.e. the *smallest* ``d`` that reaches the best count.
        Any larger, tied ``d`` would capture the exact same fraud while
        discarding strictly more mature transactions for nothing, so the
        earliest one is always the better pick, never just an arbitrary
        tie-break.
        """
        curve = self.compute_maturity_curve(max_days)
        best_idx = curve["captured_frauds"].idxmax()
        return int(curve.loc[best_idx, "date_delta_days"]), curve

    def compute_capture_milestones(self, max_days: int | None = None) -> pd.DataFrame:
        """Earliest day each new best "captured frauds" count was first reached.

        Skims the maturity curve down to the days that actually mattered:
        every time a wider window captures more fraud than any narrower
        window has so far. Reading down the resulting table shows the
        marginal cost, in transactions dropped, of each extra fraud
        captured, and that cost is rarely constant. Chasing the very last
        case can cost far more mature data than the earlier ones did, which
        is a trade-off worth looking at explicitly rather than accepting
        the global optimum automatically.
        """
        curve = self.compute_maturity_curve(max_days)
        running_best = curve["captured_frauds"].cummax()
        milestones = curve[curve["captured_frauds"] == running_best].drop_duplicates(
            "captured_frauds", keep="first"
        )
        n_total = len(self.df)
        return milestones.assign(
            transactions_dropped=n_total - milestones["mature_transactions"]
        ).reset_index(drop=True)

    def plot_maturity_curve(
        self, max_days: int | None = None, title: str | None = None
    ) -> tuple[plt.Figure, int]:
        optimal_day, curve = self.get_optimal_date_delta(max_days)
        best_count = curve.loc[curve["date_delta_days"] == optimal_day, "captured_frauds"].iloc[0]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(
            curve["date_delta_days"],
            curve["captured_frauds"],
            color=self.PALETTE["primary"],
            lw=2,
            marker="o",
            ms=3,
        )
        ax.axvline(
            optimal_day,
            color=self.PALETTE["highlight"],
            ls="--",
            lw=1.5,
            label=f"Optimal date delta = {optimal_day} days ({best_count} frauds captured)",
        )
        ax.set_xlabel("Candidate date delta (days)")
        ax.set_ylabel("Matured frauds captured")
        ax.set_title(title or "Fraud captured vs. candidate observation window", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, optimal_day

    # ------------------------------------------------------------------
    # 3. Turning a date delta into a training-ready label
    # ------------------------------------------------------------------

    def apply_date_delta_window(
        self,
        date_delta_days: int,
        drop_late_fraud: bool = False,
    ) -> pd.DataFrame:
        """Re-derive labels so they are consistent with a fixed observation window.

        Two things always happen, regardless of ``drop_late_fraud``:

        1. The **immature tail** is dropped: transactions that happened
           fewer than ``date_delta_days`` days before ``reference_date``
           haven't had the full window to be reported yet, so nothing
           trustworthy can be said about their label at this window length.
           There is no safe alternative here, defaulting them to
           legitimate would mislabel whichever of them simply haven't been
           reported yet, so they are excluded rather than guessed at.
        2. Among what remains (the *mature* transactions), the window's own
           definition of fraud gets enforced: "fraud" now specifically
           means "reported within ``date_delta_days`` days of the
           transaction". A transaction that is genuinely fraud, but was
           only confirmed after the window had already closed, no longer
           satisfies that narrower definition.

        ``drop_late_fraud`` is only about how (2) is enforced, for the
        mature transactions whose confirmed reporting lag exceeds the
        window:

        - ``drop_late_fraud=False`` (default): relabel it to legitimate
          (0). This is the standard way fixed-horizon labels are built
          ("bad within N days", "converted within N days", ...) — the row
          stays, answering the narrower, time-boxed question the label is
          now defined around.
        - ``drop_late_fraud=True``: drop the row entirely instead. Nothing
          forces a transaction that is *definitely* fraud to carry a
          "legitimate" label; the trade-off is a smaller dataset in
          exchange for never contradicting a confirmed outcome.

        Both are legitimate choices, and the right one depends on what the
        deployed model actually needs to answer: "will this be confirmed
        fraud within the window" (relabel) or "is this fraud, full stop,
        using only the cases we can speak to cleanly" (drop). In practice
        this bucket tends to be small whenever the window itself was chosen
        near the peak of the maturity curve (Section 4), by construction,
        few frauds are still being missed at that point.
        """
        out = self.df.drop(columns="_date_delta_days").copy()
        delta = self.df["_date_delta_days"]

        cutoff = self.reference_date - pd.Timedelta(days=date_delta_days)
        is_mature = self.df[self.transaction_date_col] <= cutoff
        out, delta = out[is_mature].copy(), delta[is_mature]

        is_late_fraud = (out[self.label_col] == 1) & (delta > date_delta_days)
        if drop_late_fraud:
            out = out[~is_late_fraud].copy()
        else:
            out.loc[is_late_fraud, self.label_col] = 0

        return out
