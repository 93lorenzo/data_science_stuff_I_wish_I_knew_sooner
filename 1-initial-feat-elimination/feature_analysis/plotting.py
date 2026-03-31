from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class FeatureDistributionPlotter:
    """Plots feature distributions and correlation matrices for a dataset.

    Automatically detects numerical and categorical columns from the reference
    dataframe passed at construction time. All methods accept optional ``cols``
    overrides so you can focus on a subset of features.

    All plot methods return the ``matplotlib.figure.Figure`` object and do **not**
    call ``plt.show()`` — Jupyter displays the figure automatically when it is the
    last expression in a cell, and returning it lets callers compose or save figures
    as needed.

    Parameters
    ----------
    df : pd.DataFrame
        Reference dataframe used to infer column types (typically the training set).
    target_col : str
        Name of the target column to exclude from feature lists.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "target") -> None:
        self.target_col = target_col
        self.num_cols: list[str] = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != target_col
        ]
        # is_numeric_dtype / is_bool_dtype cover int/float/bool across all pandas versions;
        # everything else (object, str, StringDtype, category …) is treated as categorical.
        self.cat_cols: list[str] = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
            and c != target_col
        ]

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------

    def plot_numerical_distributions(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        cols: Optional[list[str]] = None,
        n_cols: int = 3,
        figsize: Optional[tuple[int, int]] = None,
        bins: int = 30,
        title: str = "Numerical Feature Distributions: Train vs Test",
    ) -> plt.Figure:
        """Overlay train/test histograms for each numerical feature.

        Parameters
        ----------
        df_train, df_test : pd.DataFrame
            Training and test splits.
        cols : list of str, optional
            Subset of numerical columns to plot. Defaults to all detected ones.
        n_cols : int
            Number of subplot columns in the grid.
        figsize : tuple, optional
            Figure size. Auto-computed from grid dimensions if not provided.
        bins : int
            Number of histogram bins.
        title : str
            Figure super-title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        cols = cols or self.num_cols
        n_rows = -(-len(cols) // n_cols)  # ceiling division
        figsize = figsize or (6 * n_cols, 4 * n_rows)

        # squeeze=False guarantees axes is always a 2-D array regardless of grid shape
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.hist(df_train[col].dropna(), bins=bins, alpha=0.6, label="Train",
                    color="steelblue", density=True)
            ax.hist(df_test[col].dropna(), bins=bins, alpha=0.6, label="Test",
                    color="darkorange", density=True)
            ax.set_title(col, fontsize=11)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()

        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        return fig

    def plot_categorical_distributions(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        cols: Optional[list[str]] = None,
        n_cols: int = 3,
        figsize: Optional[tuple[int, int]] = None,
        title: str = "Categorical Feature Distributions: Train vs Test",
    ) -> plt.Figure:
        """Side-by-side bar charts comparing train/test category proportions.

        Parameters
        ----------
        df_train, df_test : pd.DataFrame
            Training and test splits.
        cols : list of str, optional
            Subset of categorical columns to plot. Defaults to all detected ones.
        n_cols : int
            Number of subplot columns in the grid.
        figsize : tuple, optional
            Figure size. Auto-computed from grid dimensions if not provided.
        title : str
            Figure super-title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        cols = cols or self.cat_cols
        n_rows = -(-len(cols) // n_cols)  # ceiling division
        figsize = figsize or (6 * n_cols, 4 * n_rows)

        # squeeze=False guarantees axes is always a 2-D array regardless of grid shape
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            train_props = df_train[col].value_counts(normalize=True).sort_index()
            test_props = df_test[col].value_counts(normalize=True).sort_index()
            categories = sorted(set(train_props.index) | set(test_props.index))
            x = np.arange(len(categories))
            width = 0.35

            ax.bar(x - width / 2, [train_props.get(c, 0) for c in categories],
                   width, label="Train", color="steelblue", alpha=0.8)
            ax.bar(x + width / 2, [test_props.get(c, 0) for c in categories],
                   width, label="Test", color="darkorange", alpha=0.8)
            ax.set_title(col, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha="right")
            ax.set_ylabel("Proportion")
            ax.legend()

        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Correlation heatmaps
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        figsize: tuple[int, int] = (10, 8),
        fmt: str = ".2f",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        annot_size: int = 11,
    ) -> plt.Figure:
        """Render a seaborn heatmap for any square correlation matrix.

        Parameters
        ----------
        corr_matrix : pd.DataFrame
            Square matrix (e.g. Pearson, Spearman, PhiK, significance).
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        fmt : str
            Annotation number format (e.g. ``'.2f'``).
        vmin, vmax : float, optional
            Colour-scale bounds. Useful for PhiK (0–1) or significance matrices.
        annot_size : int
            Font size for cell annotations.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt=fmt, cmap="Blues",
                    annot_kws={"size": annot_size}, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(title, fontsize=13)
        fig.tight_layout()
        return fig

    def plot_comparison_heatmaps(
        self,
        matrix_a: pd.DataFrame,
        matrix_b: pd.DataFrame,
        title_a: str = "Matrix A",
        title_b: str = "Matrix B",
        figsize: Optional[tuple[int, int]] = None,
        fmt: str = ".2f",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        annot_size: int = 10,
        suptitle: str = "",
    ) -> plt.Figure:
        """Render two correlation matrices side by side for direct comparison.

        Useful for comparing e.g. Pearson vs Spearman, or before vs after a
        feature-selection step.

        Parameters
        ----------
        matrix_a, matrix_b : pd.DataFrame
            The two square matrices to compare.
        title_a, title_b : str
            Subplot titles.
        figsize : tuple, optional
            Figure size. Defaults to (20, 8).
        fmt, vmin, vmax, annot_size
            Passed through to both heatmaps.
        suptitle : str
            Optional figure-level title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        figsize = figsize or (20, 8)
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

        for ax, matrix, title in [(ax_a, matrix_a, title_a), (ax_b, matrix_b, title_b)]:
            sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues",
                        annot_kws={"size": annot_size}, vmin=vmin, vmax=vmax, ax=ax)
            ax.set_title(title, fontsize=13)

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)

        fig.tight_layout()
        return fig

    def plot_feature_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        figsize: tuple[int, int] = (8, 6),
        alpha: float = 0.4,
        point_size: int = 20,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Scatter plot of two features with Pearson and Spearman r annotated in the title.

        Useful for showing *why* the two metrics diverge: a sinusoidal or other
        non-linear monotonic relationship will produce a low Pearson r but a high
        Spearman ρ, which is immediately visible in the scatter.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing both columns (typically the training set).
        x_col, y_col : str
            Column names to plot on the x and y axes.
        figsize : tuple
            Figure size.
        alpha : float
            Point transparency (0–1).
        point_size : int
            Marker size in scatter plot.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from scipy.stats import pearsonr, spearmanr

        x = df[x_col].dropna()
        y = df[y_col].dropna()
        common_idx = x.index.intersection(y.index)
        x, y = x.loc[common_idx], y.loc[common_idx]

        pearson_r, _ = pearsonr(x, y)
        spearman_r, _ = spearmanr(x, y)

        owns_fig = ax is None
        if owns_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.scatter(x, y, alpha=alpha, s=point_size, color="steelblue")
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(
            f"{x_col}  vs  {y_col}\n"
            f"Pearson r = {pearson_r:.3f}   |   Spearman ρ = {spearman_r:.3f}",
            fontsize=12,
        )
        if owns_fig:
            fig.tight_layout()
        return fig

    def plot_scatter_comparison(
        self,
        df: pd.DataFrame,
        pairs: list[tuple[str, str]],
        figsize: Optional[tuple[int, int]] = None,
        alpha: float = 0.4,
        point_size: int = 20,
        suptitle: str = "",
    ) -> plt.Figure:
        """Plot multiple scatter pairs side by side for direct comparison.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing all referenced columns.
        pairs : list of (x_col, y_col) tuples
            Each pair becomes one subplot.
        figsize : tuple, optional
            Defaults to (8 * len(pairs), 6).
        alpha, point_size
            Passed through to each scatter plot.
        suptitle : str
            Optional figure-level title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        n = len(pairs)
        figsize = figsize or (8 * n, 6)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, (x_col, y_col) in zip(axes, pairs):
            self.plot_feature_scatter(df, x_col, y_col, alpha=alpha, point_size=point_size, ax=ax)

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)

        fig.tight_layout()
        return fig

    def plot_phik_heatmap(
        self,
        phik_matrix: pd.DataFrame,
        title: str = "PhiK Correlation Matrix",
        figsize: tuple[int, int] = (10, 8),
        annot_size: int = 11,
        threshold: Optional[float] = None,
    ) -> plt.Figure:
        """Render the PhiK correlation matrix as a seaborn heatmap.

        PhiK is always in [0, 1] so the colour scale is fixed accordingly.
        Optionally overlays a mask for values below ``threshold`` to draw
        attention to only the notable correlations.

        Parameters
        ----------
        phik_matrix : pd.DataFrame
            Output of ``df.phik_matrix()``.
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        annot_size : int
            Font size for cell annotations.
        threshold : float, optional
            If provided, cells with PhiK < threshold are greyed out so the
            reader's eye is drawn to the stronger associations.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        mask = phik_matrix < threshold if threshold is not None else None

        sns.heatmap(
            phik_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot_kws={"size": annot_size},
            linewidths=0.5,
            linecolor="white",
            mask=mask,
            ax=ax,
        )
        if mask is not None:
            # draw greyed-out cells for masked values
            sns.heatmap(
                phik_matrix,
                annot=True,
                fmt=".2f",
                cmap=["#f0f0f0"],
                vmin=0,
                vmax=1,
                annot_kws={"size": annot_size, "color": "#aaaaaa"},
                linewidths=0.5,
                linecolor="white",
                mask=~mask,
                cbar=False,
                ax=ax,
            )

        ax.set_title(title, fontsize=13, pad=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        return fig
