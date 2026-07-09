"""
meta_features.py

Core module for extracting meta-features (dataset "fingerprints") from tabular
datasets. These meta-features describe structural and statistical properties
of a dataset without training any machine learning model on it.

This module is intentionally dependency-light: it only relies on pandas and
numpy, since it needs to run quickly across many datasets during meta-dataset
construction (see Day 3 and Day 5 of the DataSense roadmap).
"""

import pandas as pd
import numpy as np
from typing import Optional


class MetaFeatureExtractor:
    """
    Extracts a fixed set of meta-features describing a tabular dataset.

    A "meta-feature" here means a single descriptive statistic about the
    dataset as a whole (e.g., number of rows, percentage of missing values),
    as opposed to a feature used to train a predictive model on the data
    itself.

    Parameters
    ----------
    outlier_iqr_multiplier : float, default=1.5
        Multiplier used in the IQR (interquartile range) rule to flag
        outliers. The standard statistical convention is 1.5; we expose it
        as a parameter so future experiments can adjust sensitivity without
        modifying the class's internals.
    """

    def __init__(self, outlier_iqr_multiplier: float = 1.5):
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def extract(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> dict:
        """
        Compute all meta-features for a given dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to analyze.
        target_column : str, optional
            Name of the target/label column, if this dataset is intended
            for supervised learning. If None, target-related meta-features
            (class balance, number of classes) are skipped.

        Returns
        -------
        dict
            A flat dictionary mapping meta-feature names to their computed
            values. This flat structure is deliberate: it maps directly to
            one row of a future meta-dataset (Day 5).
        """
        # Separate the target column from the feature columns, if given.
        # We do this once, up front, so every helper method below works
        # only with the feature columns and never accidentally leaks
        # target information into feature-based statistics.
        if target_column is not None and target_column in df.columns:
            feature_df = df.drop(columns=[target_column])
            target_series = df[target_column]
        else:
            feature_df = df
            target_series = None

        meta_features = {}
        meta_features.update(self._compute_basic_shape(feature_df))
        meta_features.update(self._compute_data_quality(df))  # duplicates/missing checked on full df
        meta_features.update(self._compute_feature_composition(feature_df))
        meta_features.update(self._compute_target_info(target_series))
        meta_features.update(self._compute_statistical_structure(feature_df))

        return meta_features

    # ------------------------------------------------------------------
    # Category 1: Basic shape
    # ------------------------------------------------------------------
    def _compute_basic_shape(self, feature_df: pd.DataFrame) -> dict:
        """Row/column counts and their ratio."""
        n_rows, n_columns = feature_df.shape

        # Guard against division by zero for a pathological empty dataset.
        rows_to_columns_ratio = n_rows / n_columns if n_columns > 0 else 0.0

        return {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "rows_to_columns_ratio": rows_to_columns_ratio,
        }

    # ------------------------------------------------------------------
    # Category 2: Data quality
    # ------------------------------------------------------------------
    def _compute_data_quality(self, df: pd.DataFrame) -> dict:
        """Missing value percentage and duplicate row percentage."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_values_percentage = (
            (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
        )

        n_rows = df.shape[0]
        duplicate_rows = df.duplicated().sum()
        duplicate_rows_percentage = (
            (duplicate_rows / n_rows) * 100 if n_rows > 0 else 0.0
        )

        return {
            "missing_values_percentage": missing_values_percentage,
            "duplicate_rows_percentage": duplicate_rows_percentage,
        }

    # ------------------------------------------------------------------
    # Category 3: Feature composition
    # ------------------------------------------------------------------
    def _compute_feature_composition(self, feature_df: pd.DataFrame) -> dict:
        """Counts of numeric vs categorical columns, and their ratio."""
        numeric_columns = feature_df.select_dtypes(include=np.number).columns
        categorical_columns = feature_df.select_dtypes(exclude=np.number).columns

        n_numeric = len(numeric_columns)
        n_categorical = len(categorical_columns)

        # Avoid division by zero when a dataset has no categorical columns.
        numeric_to_categorical_ratio = (
            n_numeric / n_categorical if n_categorical > 0 else float(n_numeric)
        )

        return {
            "n_numeric_features": n_numeric,
            "n_categorical_features": n_categorical,
            "numeric_to_categorical_ratio": numeric_to_categorical_ratio,
        }

    # ------------------------------------------------------------------
    # Category 4: Target / class information
    # ------------------------------------------------------------------
    def _compute_target_info(self, target_series: Optional[pd.Series]) -> dict:
        """
        Class balance and number of classes, when a target column is given.

        class_balance is defined as (minority class count / majority class
        count), so a value of 1.0 means perfectly balanced classes, and
        values approaching 0.0 mean severe imbalance.
        """
        if target_series is None:
            return {
                "n_classes": None,
                "class_balance": None,
            }

        class_counts = target_series.value_counts()
        n_classes = len(class_counts)

        if n_classes < 2:
            class_balance = None  # Not meaningful with fewer than 2 classes.
        else:
            class_balance = class_counts.min() / class_counts.max()

        return {
            "n_classes": n_classes,
            "class_balance": class_balance,
        }

    # ------------------------------------------------------------------
    # Category 5: Statistical structure
    # ------------------------------------------------------------------
    def _compute_statistical_structure(self, feature_df: pd.DataFrame) -> dict:
        """Mean pairwise correlation, mean skewness, and outlier percentage."""
        numeric_df = feature_df.select_dtypes(include=np.number)

        mean_feature_correlation = self._compute_mean_correlation(numeric_df)
        mean_skewness = self._compute_mean_skewness(numeric_df)
        outlier_percentage = self._compute_outlier_percentage(numeric_df)

        return {
            "mean_feature_correlation": mean_feature_correlation,
            "mean_skewness": mean_skewness,
            "outlier_percentage": outlier_percentage,
        }

    def _compute_mean_correlation(self, numeric_df: pd.DataFrame) -> Optional[float]:
        """
        Average absolute pairwise correlation between numeric features.

        We need at least 2 numeric columns for correlation to be meaningful.
        We take the upper triangle of the correlation matrix (excluding the
        diagonal, which is always 1.0) to avoid double-counting each pair.
        """
        if numeric_df.shape[1] < 2:
            return None

        corr_matrix = numeric_df.corr().abs()
        # np.triu with k=1 masks everything on/below the diagonal, leaving
        # only the upper-triangle pairwise values (each pair counted once).
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        mean_correlation = upper_triangle.stack().mean()
        return float(mean_correlation) if not np.isnan(mean_correlation) else None

    def _compute_mean_skewness(self, numeric_df: pd.DataFrame) -> Optional[float]:
        """
        Average absolute skewness across numeric columns.

        We use absolute value because we care about "how non-symmetric is
        this distribution, on average", not the direction of the skew.
        Without abs(), a positively-skewed column and a negatively-skewed
        column could cancel out and hide a genuinely skewed dataset.
        """
        if numeric_df.shape[1] == 0:
            return None

        skew_values = numeric_df.skew().abs()
        mean_skew = skew_values.mean()
        return float(mean_skew) if not np.isnan(mean_skew) else None

    def _compute_outlier_percentage(self, numeric_df: pd.DataFrame) -> Optional[float]:
        """
        Percentage of numeric values considered outliers via the IQR rule.

        A value is an outlier if it falls below Q1 - k*IQR or above
        Q3 + k*IQR, where k is self.outlier_iqr_multiplier (default 1.5,
        the standard statistical convention).
        """
        if numeric_df.shape[1] == 0:
            return None

        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.outlier_iqr_multiplier * iqr
        upper_bound = q3 + self.outlier_iqr_multiplier * iqr

        is_outlier = (numeric_df < lower_bound) | (numeric_df > upper_bound)

        total_values = numeric_df.shape[0] * numeric_df.shape[1]
        outlier_count = is_outlier.sum().sum()

        return (outlier_count / total_values) * 100 if total_values > 0 else 0.0