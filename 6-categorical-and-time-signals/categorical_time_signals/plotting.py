from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CategoricalTimeSignalPlotter:
    """Plots for reading a binary label's signal out of categorical and time columns."""

    PALETTE = {
        "primary": "#5b8dd9",
        "positive": "#e05c5c",
        "neutral": "#888888",
        "highlight": "#f5a623",
        "grid": "#cccccc",
    }

    # ------------------------------------------------------------------
    # Categorical signal (nominal or ordinal)
    # ------------------------------------------------------------------

    def plot_category_rate(
        self,
        df: pd.DataFrame,
        category_col: str,
        label_col: str,
        order: Sequence[str] | None = None,
        title: str | None = None,
        overall_df: pd.DataFrame | None = None,
    ) -> plt.Figure:
        """Bar plot of the label's average rate per category.

        Pass ``order`` for an ordinal category to keep its natural order on
        the x-axis, the trend (or lack of one) along that order is the
        point. Leave it out for a nominal category and the bars are sorted
        by rate instead, there is no natural order to preserve.

        The dashed line marks the overall rate each bar is colored against.
        By default that's ``df``'s own average, but when ``df`` is itself a
        filtered slice (e.g. credit-card trips only), that average is *not*
        the rate across all data, it's the rate across the slice. Pass the
        unfiltered table as ``overall_df`` to compare each category against
        the true all-data rate instead.
        """
        rates = df.groupby(category_col, observed=True)[label_col].agg(["mean", "count"])
        rates = rates.loc[list(order)] if order is not None else rates.sort_values("mean", ascending=False)
        baseline = overall_df[label_col].mean() if overall_df is not None else df[label_col].mean()
        baseline_label = "All-data rate" if overall_df is not None else "Overall rate"

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [
            self.PALETTE["positive"] if v >= baseline else self.PALETTE["primary"]
            for v in rates["mean"]
        ]
        ax.bar(rates.index.astype(str), rates["mean"], color=colors, edgecolor="white")
        ax.axhline(
            baseline,
            color=self.PALETTE["neutral"],
            ls="--",
            lw=1.5,
            label=f"{baseline_label} = {baseline:.1%}",
        )
        for i, (rate, n) in enumerate(zip(rates["mean"], rates["count"])):
            ax.text(i, rate, f"n={n:,}", ha="center", va="bottom", fontsize=8, color=self.PALETTE["neutral"])
        ax.set_ylabel(f"{label_col} rate")
        ax.set_title(title or f"{label_col} rate by {category_col}", fontsize=12)
        y_max = max(rates["mean"].max(), baseline)
        ax.set_ylim(0, max(y_max * 1.2, 0.05))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Hour of day: the naive (linear) view
    # ------------------------------------------------------------------

    def plot_hour_rate_linear(
        self, df: pd.DataFrame, label_col: str, hour_col: str = "hour", title: str | None = None
    ) -> plt.Figure:
        """Bar plot of the label's rate across raw hour-of-day, 0 through 23."""
        rates = df.groupby(hour_col)[label_col].mean().reindex(range(24))
        baseline = df[label_col].mean()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(rates.index, rates.values, color=self.PALETTE["primary"], edgecolor="white")
        ax.axhline(
            baseline,
            color=self.PALETTE["neutral"],
            ls="--",
            lw=1.5,
            label=f"Overall rate = {baseline:.1%}",
        )
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of day (0-23)")
        ax.set_ylabel(f"{label_col} rate")
        ax.set_title(title or "Rate by raw hour-of-day", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Hour of day: the cyclical (clock) view
    # ------------------------------------------------------------------

    def plot_hour_clock(
        self, df: pd.DataFrame, label_col: str, hour_col: str = "hour", title: str | None = None
    ) -> plt.Figure:
        """Same rates as ``plot_hour_rate_linear``, arranged around a 24-hour clock."""
        rates = df.groupby(hour_col)[label_col].mean().reindex(range(24))
        angles = 2 * np.pi * rates.index / 24

        fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        sizes = 150 + 3500 * (rates.values - rates.values.min())
        sc = ax.scatter(
            angles, [1] * 24, c=rates.values, cmap="coolwarm", s=sizes, edgecolor="white", zorder=3
        )
        for h in range(24):
            ax.text(angles[h], 1.3, f"{h}", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1.45)
        ax.spines["polar"].set_visible(False)
        ax.set_title(title or "Rate by hour, arranged on a clock", fontsize=12, pad=20)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.12)
        cbar.set_label(f"{label_col} rate")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # The didactic "why" figure
    # ------------------------------------------------------------------

    def plot_hour_distance_illustration(
        self, hour_a: int = 1, hour_b: int = 23, period: int = 24
    ) -> plt.Figure:
        """Side by side: raw-hour distance (a line) vs. sin/cos distance (a clock)."""
        angle_a, angle_b = 2 * np.pi * hour_a / period, 2 * np.pi * hour_b / period

        fig, (ax_line, ax_circle) = plt.subplots(1, 2, figsize=(11, 5.2))

        # -- number line --
        ax_line.hlines(0, 0, period - 1, color=self.PALETTE["grid"], lw=2)
        ax_line.scatter([hour_a, hour_b], [0, 0], s=160, color=self.PALETTE["positive"], zorder=5)
        ax_line.annotate(
            "",
            xy=(hour_b, 0.05),
            xytext=(hour_a, 0.05),
            arrowprops={"arrowstyle": "<->", "color": self.PALETTE["positive"], "lw": 2},
        )
        ax_line.text(
            (hour_a + hour_b) / 2, 0.12, f"raw distance = {abs(hour_b - hour_a)}", ha="center", fontsize=11
        )
        for h in (hour_a, hour_b):
            ax_line.text(h, -0.14, f"{h:02d}:00", ha="center", fontsize=10)
        ax_line.set_xlim(-1, period)
        ax_line.set_ylim(-0.3, 0.3)
        ax_line.axis("off")
        ax_line.set_title("Raw hour on a number line", fontsize=12)

        # -- clock circle --
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_circle.plot(np.cos(theta), np.sin(theta), color=self.PALETTE["grid"], lw=1.5)
        for h in range(period):
            a = 2 * np.pi * h / period
            ax_circle.scatter([np.cos(a)], [np.sin(a)], color=self.PALETTE["grid"], s=20, zorder=3)
        for h in (hour_a, hour_b):
            a = 2 * np.pi * h / period
            ax_circle.scatter([np.cos(a)], [np.sin(a)], color=self.PALETTE["positive"], s=160, zorder=5)
            ax_circle.text(1.2 * np.cos(a), 1.2 * np.sin(a), f"{h:02d}:00", ha="center", va="center", fontsize=10)
        cyclical_distance = np.hypot(np.cos(angle_a) - np.cos(angle_b), np.sin(angle_a) - np.sin(angle_b))
        ax_circle.plot(
            [np.cos(angle_a), np.cos(angle_b)],
            [np.sin(angle_a), np.sin(angle_b)],
            color=self.PALETTE["positive"],
            lw=2,
            ls="--",
        )
        ax_circle.text(0, -1.3, f"sin/cos distance = {cyclical_distance:.2f}", ha="center", fontsize=11)
        ax_circle.set_xlim(-1.5, 1.5)
        ax_circle.set_ylim(-1.5, 1.5)
        ax_circle.set_aspect("equal")
        ax_circle.axis("off")
        ax_circle.set_title("sin/cos coordinates on a clock", fontsize=12)

        fig.suptitle(
            f"Hour {hour_a:02d}:00 and hour {hour_b:02d}:00: far on a line, close on a clock", fontsize=13
        )
        fig.tight_layout()
        return fig
