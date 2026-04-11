from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
)


class ClassifierMetricsPlotter:
    """Plotting utilities for binary classifier evaluation."""

    # Colour palette shared across methods
    PALETTE = {
        "positive": "#e05c5c",
        "negative": "#5b8dd9",
        "neutral": "#888888",
        "green": "#5cb85c",
        "amber": "#f5a623",
        "red": "#e05c5c",
        "highlight": "#f9a825",
    }

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        labels: tuple[str, str] = ("Legitimate", "Fraud"),
        title: str | None = None,
    ) -> plt.Figure:
        """Heatmap confusion matrix at a given decision threshold."""
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Pred {l}" for l in labels],
            yticklabels=[f"True {l}" for l in labels],
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
        )
        ax.set_title(title or f"Confusion matrix  (threshold = {threshold:.2f})", fontsize=12)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Precision / Recall vs threshold
    # ------------------------------------------------------------------

    def plot_pr_vs_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        highlight_threshold: float | None = None,
        title: str = "Precision & Recall vs decision threshold",
    ) -> plt.Figure:
        """Line chart showing precision and recall as the decision threshold varies."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        # precision_recall_curve returns one extra point for precision/recall
        thresholds_ext = np.append(thresholds, 1.0)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresholds_ext, precisions, label="Precision", color=self.PALETTE["negative"], lw=2)
        ax.plot(thresholds_ext, recalls, label="Recall", color=self.PALETTE["positive"], lw=2)
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        if highlight_threshold is not None:
            idx = np.searchsorted(thresholds, highlight_threshold)
            p_val = precisions[idx]
            r_val = recalls[idx]
            ax.axvline(highlight_threshold, color=self.PALETTE["highlight"], ls="--", lw=1.5,
                       label=f"threshold = {highlight_threshold:.2f}")
            ax.scatter([highlight_threshold, highlight_threshold], [p_val, r_val],
                       color=self.PALETTE["highlight"], zorder=5)
            ax.legend()

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        max_fpr: float | None = None,
        title: str | None = None,
    ) -> plt.Figure:
        """ROC curve, optionally shading the partial-AUC region up to max_fpr."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        full_auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=self.PALETTE["negative"], lw=2,
                label=f"ROC AUC = {full_auc:.3f}")
        ax.plot([0, 1], [0, 1], color=self.PALETTE["neutral"], ls="--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "ROC curve", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

        if max_fpr is not None:
            partial_auc = roc_auc_score(y_true, y_proba, max_fpr=max_fpr)
            # Shade the partial region
            mask = fpr <= max_fpr
            fpr_region = np.append(fpr[mask], max_fpr)
            # interpolate tpr at max_fpr
            tpr_at_max = np.interp(max_fpr, fpr, tpr)
            tpr_region = np.append(tpr[mask], tpr_at_max)
            ax.fill_between(fpr_region, 0, tpr_region,
                            alpha=0.25, color=self.PALETTE["amber"],
                            label=f"Partial AUC (FPR ≤ {max_fpr}) = {partial_auc:.3f}")
            ax.axvline(max_fpr, color=self.PALETTE["amber"], ls="--", lw=1.5)

        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Precision-Recall curve
    # ------------------------------------------------------------------

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall curve",
    ) -> plt.Figure:
        """PR curve with AUC annotation and random-classifier baseline."""
        precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recalls, precisions)
        baseline = y_true.mean()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recalls, precisions, color=self.PALETTE["positive"], lw=2,
                label=f"PR AUC = {pr_auc:.3f}")
        ax.axhline(baseline, color=self.PALETTE["neutral"], ls="--", lw=1,
                   label=f"Random baseline (prevalence = {baseline:.2%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Side-by-side ROC + PR
    # ------------------------------------------------------------------

    def plot_roc_and_pr(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC and Precision-Recall curves",
    ) -> plt.Figure:
        """ROC and PR curves side by side."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = auc(recalls, precisions)
        baseline = y_true.mean()

        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(title, fontsize=13)

        # ROC
        ax_roc.plot(fpr, tpr, color=self.PALETTE["negative"], lw=2,
                    label=f"ROC AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], color=self.PALETTE["neutral"], ls="--", lw=1, label="Random")
        ax_roc.fill_between(fpr, 0, tpr, alpha=0.10, color=self.PALETTE["negative"])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC curve")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.02)
        ax_roc.grid(alpha=0.3)
        ax_roc.legend(fontsize=9)

        # PR
        ax_pr.plot(recalls, precisions, color=self.PALETTE["positive"], lw=2,
                   label=f"PR AUC = {pr_auc:.3f}")
        ax_pr.axhline(baseline, color=self.PALETTE["neutral"], ls="--", lw=1,
                      label=f"Baseline = {baseline:.2%}")
        ax_pr.fill_between(recalls, baseline, precisions, alpha=0.10, color=self.PALETTE["positive"])
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall curve")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.05)
        ax_pr.grid(alpha=0.3)
        ax_pr.legend(fontsize=9)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Business cost curve
    # ------------------------------------------------------------------

    def plot_cost_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fn_cost: float,
        fp_cost: float,
        title: str | None = None,
    ) -> plt.Figure:
        """Total business cost as a function of the decision threshold.

        Parameters
        ----------
        fn_cost : float
            Cost (euros) of one false negative (missed fraud).
        fp_cost : float
            Cost (euros) of one false positive (false alarm investigated).
        """
        _, _, thresholds = roc_curve(y_true, y_proba)

        total_costs, fn_costs, fp_costs = [], [], []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            c_fn = fn * fn_cost
            c_fp = fp * fp_cost
            total_costs.append(c_fn + c_fp)
            fn_costs.append(c_fn)
            fp_costs.append(c_fp)

        total_costs = np.array(total_costs)
        best_idx = np.argmin(total_costs)
        best_threshold = thresholds[best_idx]
        best_cost = total_costs[best_idx]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(thresholds, np.array(total_costs) / 1_000, color="black", lw=2, label="Total cost")
        ax.plot(thresholds, np.array(fn_costs) / 1_000, color=self.PALETTE["red"],
                ls="--", lw=1.5, label=f"FN cost  ({fn_cost:,.0f} €/miss)")
        ax.plot(thresholds, np.array(fp_costs) / 1_000, color=self.PALETTE["negative"],
                ls="--", lw=1.5, label=f"FP cost  ({fp_cost:,.0f} €/alarm)")
        ax.axvline(best_threshold, color=self.PALETTE["amber"], ls="--", lw=1.5,
                   label=f"Optimal threshold = {best_threshold:.2f}  ({best_cost / 1_000:,.1f}k €)")
        ax.scatter([best_threshold], [best_cost / 1_000],
                   color=self.PALETTE["amber"], zorder=5, s=80)
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Cost (thousands €)")
        ax.set_title(title or "Business cost vs decision threshold", fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig, best_threshold

    # ------------------------------------------------------------------
    # Partial ROC AUC
    # ------------------------------------------------------------------

    def plot_partial_roc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        max_fpr: float = 0.10,
        title: str | None = None,
    ) -> plt.Figure:
        """ROC curve with partial-AUC region highlighted and capacity annotation."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        full_auc = roc_auc_score(y_true, y_proba)
        partial_auc = roc_auc_score(y_true, y_proba, max_fpr=max_fpr)

        tpr_at_max = np.interp(max_fpr, fpr, tpr)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        # Full curve (greyed)
        ax.plot(fpr, tpr, color=self.PALETTE["neutral"], lw=1.5, alpha=0.5,
                label=f"Full ROC AUC = {full_auc:.3f}")
        # Partial region (coloured)
        mask = fpr <= max_fpr
        fpr_region = np.append(fpr[mask], max_fpr)
        tpr_region = np.append(tpr[mask], tpr_at_max)
        ax.plot(fpr_region, tpr_region, color=self.PALETTE["negative"], lw=2.5,
                label=f"Partial ROC AUC (FPR ≤ {max_fpr:.0%}) = {partial_auc:.3f}")
        ax.fill_between(fpr_region, 0, tpr_region, alpha=0.20, color=self.PALETTE["negative"])

        # Capacity line
        ax.axvline(max_fpr, color=self.PALETTE["amber"], ls="--", lw=1.5,
                   label=f"Capacity limit: FPR = {max_fpr:.0%}")
        ax.scatter([max_fpr], [tpr_at_max], color=self.PALETTE["amber"], zorder=5, s=80)

        # Random baseline
        ax.plot([0, max_fpr], [0, max_fpr], color=self.PALETTE["neutral"], ls=":", lw=1,
                label="Random in region")

        ax.set_xlabel("False Positive Rate  (fraction of legitimate transactions alerted)")
        ax.set_ylabel("True Positive Rate  (recall)")
        ax.set_title(title or f"Partial ROC AUC — focusing on FPR ≤ {max_fpr:.0%}", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig
