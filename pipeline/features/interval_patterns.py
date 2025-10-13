"""Interval Pattern Feature Engineering

This module contains the IntervalPatternFeatureEngine class for creating
features based on interval (tier/segment) changes over time.

The dataset contains binned/interval features like RC_M1_SAA (sales intervals):
- 1: Top 10% (best performers)
- 2: 10-25%
- 3: 25-50%
- 4: 50-75%
- 5: 75-90%
- 6: Bottom 10% (worst performers)

Key insight: Tracking interval degradation is a strong signal for business decline.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class IntervalPatternFeatureEngine:
    """
    Interval Pattern Feature Engineering class.

    Creates features based on interval changes:
    - Interval decline patterns
    - Consecutive decline tracking
    - Decline speed/acceleration
    - Historical worst indicators
    - Cross-metric interval analysis
    """

    def __init__(self, merchant_col: str = 'ENCODED_MCT', date_col: str = 'TA_YM'):
        """
        Initialize IntervalPatternFeatureEngine.

        Args:
            merchant_col: Column name for merchant ID
            date_col: Column name for date (YYYYMM format)
        """
        self.merchant_col = merchant_col
        self.date_col = date_col

    def create_interval_decline_features(
        self,
        df: pd.DataFrame,
        interval_columns: List[str],
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create interval decline pattern features.

        Features include:
        - Number of declines in window (interval increased = worse performance)
        - Consecutive decline count
        - Decline speed (interval change magnitude)
        - Recovery indicators (improvement after decline)

        Args:
            df: Input DataFrame
            interval_columns: List of interval columns (e.g., RC_M1_SAA, RC_M1_TO_UE_CT)
            windows: List of window sizes to calculate decline counts

        Returns:
            DataFrame with interval decline features added
        """
        print(f"\nCreating interval decline features...")
        print(f"Columns: {len(interval_columns)}")
        print(f"Windows: {windows}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        features_created = 0

        for col in interval_columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            # 1. Month-over-month interval change (positive = decline, negative = improvement)
            interval_change_col = f"{col}_interval_change"
            df_result[interval_change_col] = df_result.groupby(self.merchant_col)[col].diff()
            features_created += 1

            # 2. Decline flag (interval increased = worse)
            decline_flag_col = f"{col}_is_declining"
            df_result[decline_flag_col] = (df_result[interval_change_col] > 0).astype(int)
            features_created += 1

            # 3. Consecutive decline count
            consecutive_decline_col = f"{col}_consecutive_declines"
            df_result[consecutive_decline_col] = df_result.groupby(self.merchant_col)[decline_flag_col].apply(
                lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
            ).reset_index(level=0, drop=True)
            features_created += 1

            # 4. Decline count within windows
            for window in windows:
                decline_count_col = f"{col}_decline_count_{window}m"
                df_result[decline_count_col] = df_result.groupby(self.merchant_col)[decline_flag_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
                features_created += 1

            # 5. Total interval decline from N months ago
            for window in [3, 6, 12]:
                total_decline_col = f"{col}_total_decline_{window}m"
                df_result[total_decline_col] = df_result.groupby(self.merchant_col)[col].diff(periods=window)
                features_created += 1

            # 6. Decline speed (average interval decline per month)
            for window in [3, 6]:
                decline_speed_col = f"{col}_decline_speed_{window}m"
                total_decline = df_result.groupby(self.merchant_col)[col].diff(periods=window)
                df_result[decline_speed_col] = total_decline / window
                features_created += 1

        print(f"Created {features_created} interval decline features")

        return df_result

    def create_historical_worst_features(
        self,
        df: pd.DataFrame,
        interval_columns: List[str]
    ) -> pd.DataFrame:
        """
        Create historical worst interval indicators.

        Features include:
        - Worst interval ever reached
        - Is currently at worst
        - Months since worst
        - Distance from best interval

        Args:
            df: Input DataFrame
            interval_columns: List of interval columns

        Returns:
            DataFrame with historical worst features added
        """
        print(f"\nCreating historical worst features...")
        print(f"Columns: {len(interval_columns)}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        features_created = 0

        for col in interval_columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found, skipping...")
                continue

            # 1. Worst interval ever (maximum value = worst performance)
            worst_ever_col = f"{col}_worst_ever"
            df_result[worst_ever_col] = df_result.groupby(self.merchant_col)[col].cummax()
            features_created += 1

            # 2. Best interval ever (minimum value = best performance)
            best_ever_col = f"{col}_best_ever"
            df_result[best_ever_col] = df_result.groupby(self.merchant_col)[col].cummin()
            features_created += 1

            # 3. Is at worst now (boolean)
            at_worst_col = f"{col}_at_worst_now"
            df_result[at_worst_col] = (df_result[col] == df_result[worst_ever_col]).astype(int)
            features_created += 1

            # 4. Distance from best
            distance_from_best_col = f"{col}_distance_from_best"
            df_result[distance_from_best_col] = df_result[col] - df_result[best_ever_col]
            features_created += 1

            # 5. Months since best performance
            def months_since_best(group):
                """Calculate months since best interval."""
                result = pd.Series(index=group.index, dtype=float)
                best_val = group[best_ever_col].iloc[0]
                months_counter = 0

                for idx in group.index:
                    current_val = group.loc[idx, col]
                    if current_val == best_val:
                        months_counter = 0
                    else:
                        months_counter += 1
                    result.loc[idx] = months_counter

                return result

            months_since_best_col = f"{col}_months_since_best"
            df_result[months_since_best_col] = df_result.groupby(self.merchant_col).apply(
                months_since_best
            ).reset_index(level=0, drop=True)
            features_created += 1

        print(f"Created {features_created} historical worst features")

        return df_result

    def create_recovery_indicators(
        self,
        df: pd.DataFrame,
        interval_columns: List[str]
    ) -> pd.DataFrame:
        """
        Create recovery/improvement indicators after decline.

        Features include:
        - Recovery flag (improved after decline)
        - Recovery speed
        - Failed recovery attempts
        - Volatile interval movement

        Args:
            df: Input DataFrame
            interval_columns: List of interval columns

        Returns:
            DataFrame with recovery indicator features added
        """
        print(f"\nCreating recovery indicators...")
        print(f"Columns: {len(interval_columns)}")

        df_result = df.copy()

        # Sort by merchant and date
        df_result = df_result.sort_values([self.merchant_col, self.date_col])

        features_created = 0

        for col in interval_columns:
            interval_change_col = f"{col}_interval_change"

            if interval_change_col not in df_result.columns:
                print(f"Warning: Change column '{interval_change_col}' not found, skipping...")
                continue

            # 1. Recovery flag (negative change = improvement)
            recovery_flag_col = f"{col}_is_recovering"
            df_result[recovery_flag_col] = (df_result[interval_change_col] < 0).astype(int)
            features_created += 1

            # 2. Consecutive recovery count
            consecutive_recovery_col = f"{col}_consecutive_recovery"
            df_result[consecutive_recovery_col] = df_result.groupby(self.merchant_col)[recovery_flag_col].apply(
                lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
            ).reset_index(level=0, drop=True)
            features_created += 1

            # 3. Recovery after decline pattern (decline -> improvement)
            decline_flag_col = f"{col}_is_declining"
            if decline_flag_col in df_result.columns:
                recovery_after_decline_col = f"{col}_recovery_after_decline"
                prev_decline = df_result.groupby(self.merchant_col)[decline_flag_col].shift(1)
                df_result[recovery_after_decline_col] = (
                    (prev_decline == 1) & (df_result[recovery_flag_col] == 1)
                ).astype(int)
                features_created += 1

            # 4. Interval volatility (frequent ups and downs = instability)
            for window in [3, 6]:
                volatility_col = f"{col}_interval_volatility_{window}m"
                df_result[volatility_col] = df_result.groupby(self.merchant_col)[interval_change_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                features_created += 1

            # 5. Direction change count (oscillation indicator)
            for window in [3, 6]:
                direction_change_col = f"{col}_direction_changes_{window}m"
                # Calculate sign change (direction reversal)
                current_change = df_result[interval_change_col]
                prev_change = df_result.groupby(self.merchant_col)[interval_change_col].shift(1)
                sign_change = (current_change * prev_change < 0).astype(int)

                # Store as temporary column to use in rolling
                temp_col = f"_temp_sign_change_{col}"
                df_result[temp_col] = sign_change

                # Calculate rolling sum
                df_result[direction_change_col] = df_result.groupby(self.merchant_col)[temp_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )

                # Drop temporary column
                df_result.drop(columns=[temp_col], inplace=True)
                features_created += 1

        print(f"Created {features_created} recovery indicator features")

        return df_result

    def create_cross_metric_interval_features(
        self,
        df: pd.DataFrame,
        primary_col: str = 'RC_M1_SAA',
        secondary_cols: List[str] = ['RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN']
    ) -> pd.DataFrame:
        """
        Create cross-metric interval comparison features.

        Identifies divergence patterns like:
        - Sales interval declining but customer count stable (price issue)
        - Customer count declining but sales stable (dependency on few customers)

        Args:
            df: Input DataFrame
            primary_col: Primary interval column (usually sales)
            secondary_cols: Secondary interval columns to compare

        Returns:
            DataFrame with cross-metric features added
        """
        print(f"\nCreating cross-metric interval features...")
        print(f"Primary: {primary_col}")
        print(f"Secondary: {secondary_cols}")

        df_result = df.copy()

        features_created = 0

        primary_change_col = f"{primary_col}_interval_change"

        if primary_change_col not in df_result.columns:
            print(f"Warning: Primary change column '{primary_change_col}' not found")
            return df_result

        for sec_col in secondary_cols:
            sec_change_col = f"{sec_col}_interval_change"

            if sec_change_col not in df_result.columns:
                print(f"Warning: Secondary change column '{sec_change_col}' not found, skipping...")
                continue

            # 1. Divergence: primary declining but secondary stable/improving
            divergence_col = f"divergence_{primary_col}_vs_{sec_col}"
            df_result[divergence_col] = (
                (df_result[primary_change_col] > 0) &  # Sales declining
                (df_result[sec_change_col] <= 0)        # Customer stable/improving
            ).astype(int)
            features_created += 1

            # 2. Aligned decline (both declining = critical)
            aligned_decline_col = f"aligned_decline_{primary_col}_{sec_col}"
            df_result[aligned_decline_col] = (
                (df_result[primary_change_col] > 0) &
                (df_result[sec_change_col] > 0)
            ).astype(int)
            features_created += 1

            # 3. Divergence magnitude
            divergence_magnitude_col = f"divergence_magnitude_{primary_col}_{sec_col}"
            df_result[divergence_magnitude_col] = (
                df_result[primary_change_col] - df_result[sec_change_col]
            )
            features_created += 1

        print(f"Created {features_created} cross-metric interval features")

        return df_result

    def create_all_interval_features(
        self,
        df: pd.DataFrame,
        interval_columns: List[str] = ['RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT']
    ) -> pd.DataFrame:
        """
        Create all interval pattern features at once.

        This is a convenience method that calls all feature creation methods.

        Args:
            df: Input DataFrame
            interval_columns: List of interval columns to process

        Returns:
            DataFrame with all interval pattern features added
        """
        print("\n" + "="*80)
        print("CREATING ALL INTERVAL PATTERN FEATURES")
        print("="*80)

        # Step 1: Decline features
        df_result = self.create_interval_decline_features(df, interval_columns)

        # Step 2: Historical worst features
        df_result = self.create_historical_worst_features(df_result, interval_columns)

        # Step 3: Recovery indicators
        df_result = self.create_recovery_indicators(df_result, interval_columns)

        # Step 4: Cross-metric features (using first column as primary)
        if len(interval_columns) > 1:
            df_result = self.create_cross_metric_interval_features(
                df_result,
                primary_col=interval_columns[0],
                secondary_cols=interval_columns[1:]
            )

        total_new_features = df_result.shape[1] - df.shape[1]
        print(f"\n{'='*80}")
        print(f"INTERVAL PATTERN FEATURES COMPLETE")
        print(f"Total new features created: {total_new_features}")
        print(f"{'='*80}\n")

        return df_result
