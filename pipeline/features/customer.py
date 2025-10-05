"""Customer Behavior Feature Engineering

This module contains the CustomerFeatureEngine class for creating
customer behavior and loyalty features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class CustomerFeatureEngine:
    """
    Customer Behavior Feature Engineering class.

    Creates customer behavior and loyalty features:
    - Customer behavior patterns
    - Loyalty indicators
    - Retention metrics
    """

    def __init__(self, merchant_col: str = 'ENCODED_MCT', date_col: str = 'TA_YM'):
        """
        Initialize CustomerFeatureEngine.

        Args:
            merchant_col: Column name for merchant ID
            date_col: Column name for date (YYYYMM format)
        """
        self.merchant_col = merchant_col
        self.date_col = date_col

    def create_customer_behavior_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create customer behavior features.

        Features include:
        - Average returning customer ratio
        - Average new customer ratio
        - Trend in customer acquisition
        - Customer retention stability

        Args:
            df: Input DataFrame
            windows: List of window sizes (in months)

        Returns:
            DataFrame with customer behavior features added
        """
        print(f"\nCreating customer behavior features...")
        print(f"Windows: {windows}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        # Check if required columns exist
        reu_col = 'MCT_UE_CLN_REU_RAT'
        new_col = 'MCT_UE_CLN_NEW_RAT'

        if reu_col not in df_result.columns or new_col not in df_result.columns:
            print(f"Warning: Required columns not found, skipping customer behavior features")
            return df_result

        features_created = 0

        for window in windows:
            # Average returning customer ratio
            df_result[f'customer_reu_avg_{window}m'] = df_result.groupby(self.merchant_col)[reu_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features_created += 1

            # Average new customer ratio
            df_result[f'customer_new_avg_{window}m'] = df_result.groupby(self.merchant_col)[new_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features_created += 1

            # Standard deviation of returning customer ratio (stability)
            df_result[f'customer_reu_std_{window}m'] = df_result.groupby(self.merchant_col)[reu_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            features_created += 1

            # Trend in new customer acquisition
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

            df_result[f'customer_new_trend_{window}m'] = df_result.groupby(self.merchant_col)[new_col].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(calculate_trend, raw=False)
            )
            features_created += 1

        print(f"Created {features_created} customer behavior features")

        return df_result

    def create_loyalty_indicators(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create customer loyalty indicators.

        Features include:
        - Loyalty score (returning / new customer ratio)
        - Loyalty trend
        - Customer stability index

        Args:
            df: Input DataFrame
            windows: List of window sizes (in months)

        Returns:
            DataFrame with loyalty indicator features added
        """
        print(f"\nCreating loyalty indicators...")
        print(f"Windows: {windows}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        # Check if required columns exist
        reu_col = 'MCT_UE_CLN_REU_RAT'
        new_col = 'MCT_UE_CLN_NEW_RAT'

        if reu_col not in df_result.columns or new_col not in df_result.columns:
            print(f"Warning: Required columns not found, skipping loyalty indicators")
            return df_result

        features_created = 0

        # Instant loyalty score (returning / new ratio)
        loyalty_score = df_result[reu_col] / df_result[new_col].replace(0, np.nan)
        # Replace inf values with NaN (occurs when dividing by 0 or very small values)
        loyalty_score = loyalty_score.replace([np.inf, -np.inf], np.nan)
        df_result['loyalty_score'] = loyalty_score
        features_created += 1

        for window in windows:
            # Average loyalty score
            df_result[f'loyalty_score_avg_{window}m'] = df_result.groupby(self.merchant_col)['loyalty_score'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features_created += 1

            # Loyalty trend
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

            df_result[f'loyalty_trend_{window}m'] = df_result.groupby(self.merchant_col)['loyalty_score'].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(calculate_trend, raw=False)
            )
            features_created += 1

            # Customer stability index (inverse of coefficient of variation)
            # Higher stability = lower CV
            rolling_mean = df_result.groupby(self.merchant_col)[reu_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            rolling_std = df_result.groupby(self.merchant_col)[reu_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            cv = rolling_std / rolling_mean.replace(0, np.nan)
            # Replace inf values with NaN
            cv = cv.replace([np.inf, -np.inf], np.nan)
            stability = 1 / (1 + cv)
            # Replace any remaining inf values
            stability = stability.replace([np.inf, -np.inf], np.nan)
            df_result[f'customer_stability_{window}m'] = stability
            features_created += 1

        print(f"Created {features_created} loyalty indicator features")

        return df_result
