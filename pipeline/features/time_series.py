"""Time Series Feature Engineering

This module contains the TimeSeriesFeatureEngine class for creating
time series features including lag features, moving averages, change rates,
trend indicators, volatility indicators, and ranking features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class TimeSeriesFeatureEngine:
    """
    Time Series Feature Engineering class.

    Creates various time series features for merchant data:
    - Lag features
    - Moving averages
    - Change rates
    - Trend indicators
    - Volatility indicators
    - Ranking indicators
    """

    def __init__(self, merchant_col: str = 'ENCODED_MCT', date_col: str = 'TA_YM'):
        """
        Initialize TimeSeriesFeatureEngine.

        Args:
            merchant_col: Column name for merchant ID
            date_col: Column name for date (YYYYMM format)
        """
        self.merchant_col = merchant_col
        self.date_col = date_col

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.

        Args:
            df: Input DataFrame
            columns: List of columns to create lag features for
            lags: List of lag periods (in months)

        Returns:
            DataFrame with lag features added
        """
        print(f"\nCreating lag features...")
        print(f"Columns: {len(columns)}")
        print(f"Lags: {lags}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}m"
                df_result[lag_col_name] = df_result.groupby(self.merchant_col)[col].shift(lag)

        print(f"Created {len(columns) * len(lags)} lag features")

        return df_result

    def create_moving_averages(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create moving average features.

        Args:
            df: Input DataFrame
            columns: List of columns to create moving averages for
            windows: List of window sizes (in months)

        Returns:
            DataFrame with moving average features added
        """
        print(f"\nCreating moving averages...")
        print(f"Columns: {len(columns)}")
        print(f"Windows: {windows}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            for window in windows:
                ma_col_name = f"{col}_ma_{window}m"
                df_result[ma_col_name] = df_result.groupby(self.merchant_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

        print(f"Created {len(columns) * len(windows)} moving average features")

        return df_result

    def create_change_rates(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 3, 12]
    ) -> pd.DataFrame:
        """
        Create change rate (percent change) features.

        Args:
            df: Input DataFrame
            columns: List of columns to create change rates for
            periods: List of periods to calculate change over (in months)

        Returns:
            DataFrame with change rate features added
        """
        print(f"\nCreating change rates...")
        print(f"Columns: {len(columns)}")
        print(f"Periods: {periods}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            for period in periods:
                change_col_name = f"{col}_change_{period}m"
                change_values = df_result.groupby(self.merchant_col)[col].pct_change(periods=period) * 100
                # Replace inf values with NaN (occurs when dividing by 0)
                change_values = change_values.replace([np.inf, -np.inf], np.nan)
                df_result[change_col_name] = change_values

        print(f"Created {len(columns) * len(periods)} change rate features")

        return df_result

    def create_trend_indicators(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create trend indicators (slope of linear regression).

        Args:
            df: Input DataFrame
            columns: List of columns to create trend indicators for
            windows: List of window sizes (in months)

        Returns:
            DataFrame with trend indicator features added
        """
        print(f"\nCreating trend indicators...")
        print(f"Columns: {len(columns)}")
        print(f"Windows: {windows}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        def calculate_trend(series):
            """Calculate linear trend (slope) for a series."""
            if len(series) < 2 or series.isna().all():
                return np.nan
            x = np.arange(len(series))
            y = series.values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            trend = np.polyfit(x[mask], y[mask], 1)[0]
            return trend

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            for window in windows:
                trend_col_name = f"{col}_trend_{window}m"
                df_result[trend_col_name] = df_result.groupby(self.merchant_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(calculate_trend, raw=False)
                )

        print(f"Created {len(columns) * len(windows)} trend indicator features")

        return df_result

    def create_volatility_indicators(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create volatility indicators (standard deviation and coefficient of variation).

        Args:
            df: Input DataFrame
            columns: List of columns to create volatility indicators for
            windows: List of window sizes (in months)

        Returns:
            DataFrame with volatility indicator features added
        """
        print(f"\nCreating volatility indicators...")
        print(f"Columns: {len(columns)}")
        print(f"Windows: {windows}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            for window in windows:
                # Standard deviation
                std_col_name = f"{col}_std_{window}m"
                df_result[std_col_name] = df_result.groupby(self.merchant_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )

                # Coefficient of variation (CV = std / mean)
                cv_col_name = f"{col}_cv_{window}m"
                rolling_mean = df_result.groupby(self.merchant_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                rolling_std = df_result[std_col_name]
                # Replace 0 and very small values with NaN to avoid inf
                cv_values = rolling_std / rolling_mean.replace(0, np.nan)
                cv_values = cv_values.replace([np.inf, -np.inf], np.nan)
                df_result[cv_col_name] = cv_values

        print(f"Created {len(columns) * len(windows) * 2} volatility indicator features")

        return df_result

    def create_ranking_indicators(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Create ranking indicators (rank and percentile rank within each month).

        Args:
            df: Input DataFrame
            columns: List of columns to create rankings for

        Returns:
            DataFrame with ranking features added
        """
        print(f"\nCreating ranking indicators...")
        print(f"Columns: {len(columns)}")

        df_result = df.copy()

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            # Rank (higher value = better rank = lower number)
            rank_col_name = f"{col}_rank"
            df_result[rank_col_name] = df_result.groupby(self.date_col)[col].rank(ascending=False, method='min')

            # Percentile rank (0-100)
            rank_pct_col_name = f"{col}_rank_pct"
            df_result[rank_pct_col_name] = df_result.groupby(self.date_col)[col].rank(pct=True) * 100

        print(f"Created {len(columns) * 2} ranking features")

        return df_result

    def create_ranking_change(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 3, 6]
    ) -> pd.DataFrame:
        """
        Create ranking change features (change in rank over time).

        Args:
            df: Input DataFrame
            columns: List of columns to create ranking changes for
            periods: List of periods to calculate rank change over (in months)

        Returns:
            DataFrame with ranking change features added
        """
        print(f"\nCreating ranking change indicators...")
        print(f"Columns: {len(columns)}")
        print(f"Periods: {periods}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        for col in columns:
            rank_col = f"{col}_rank"

            if rank_col not in df_result.columns:
                print(f"Warning: Rank column '{rank_col}' not found, skipping...")
                continue

            for period in periods:
                rank_change_col_name = f"{col}_rank_change_{period}m"
                # Negative change = rank improved (went down in number)
                # Positive change = rank worsened (went up in number)
                df_result[rank_change_col_name] = df_result.groupby(self.merchant_col)[rank_col].diff(periods=period)

        print(f"Created {len(columns) * len(periods)} ranking change features")

        return df_result
