"""Seaborn-based statistical charts."""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure


class SeabornCharts:
    """Generator for Seaborn statistical charts."""

    @staticmethod
    def create_heatmap(
        heatmap_data: dict,
        title: str = "Heatmap",
        figsize=(12, 8),
        cmap: str = "YlOrRd",
    ) -> Figure:
        """
        Create a heatmap using Seaborn.

        Args:
            heatmap_data: Dictionary with x_labels, y_labels, values
            title: Chart title
            figsize: Figure size
            cmap: Colormap name

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            heatmap_data["values"],
            xticklabels=heatmap_data["x_labels"],
            yticklabels=heatmap_data["y_labels"],
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar_kws={"label": "Count"},
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig

    @staticmethod
    def create_boxplot(
        df: pl.DataFrame, col: str, title: str = "Box Plot", figsize=(8, 6)
    ) -> Figure:
        """
        Create a box plot.

        Args:
            df: DataFrame with data
            col: Column name to plot
            title: Chart title
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Convert Polars to list for Seaborn
        data = df[col].to_list()

        sns.boxplot(y=data, ax=ax, color="skyblue")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        return fig

    @staticmethod
    def create_violin_plot(
        df: pl.DataFrame, col: str, title: str = "Violin Plot", figsize=(8, 6)
    ) -> Figure:
        """
        Create a violin plot.

        Args:
            df: DataFrame with data
            col: Column name to plot
            title: Chart title
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        data = df[col].to_list()

        sns.violinplot(y=data, ax=ax, color="lightblue")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        return fig

    @staticmethod
    def create_count_plot(
        df: pl.DataFrame,
        col: str,
        title: str = "Count Plot",
        figsize=(10, 6),
        top_n: int = None,
    ) -> Figure:
        """
        Create a count plot for categorical data.

        Args:
            df: DataFrame with data
            col: Column name to plot
            title: Chart title
            figsize: Figure size
            top_n: Show only top N categories

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get value counts
        value_counts = (
            df.group_by(col)
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )

        if top_n:
            value_counts = value_counts.head(top_n)

        categories = value_counts[col].to_list()
        counts = value_counts["count"].to_list()

        sns.barplot(x=categories, y=counts, ax=ax, palette="viridis")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return fig
