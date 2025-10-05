"""Composite Feature Engineering

This module contains the CompositeFeatureEngine class for creating
composite features by combining multiple indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class CompositeFeatureEngine:
    """
    Composite Feature Engineering class.

    Creates composite features by combining multiple indicators:
    - Merchant health index
    - Risk index
    - Growth index
    """

    def __init__(self):
        """Initialize CompositeFeatureEngine."""
        pass

    def create_composite_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create composite indicators.

        Composite indicators include:
        - Health index: combination of sales, customers, and loyalty
        - Risk index: combination of volatility, trend, and ranking
        - Growth index: combination of change rates and trends

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with composite indicator features added
        """
        print(f"\nCreating composite indicators...")

        df_result = df.copy()
        features_created = 0

        # Health Index (3, 6, 12 month windows)
        for window in [3, 6, 12]:
            health_components = []

            # Sales trend component
            sales_trend_col = f'RC_M1_SAA_trend_{window}m'
            if sales_trend_col in df_result.columns:
                health_components.append(sales_trend_col)

            # Customer loyalty component
            loyalty_col = f'loyalty_score_avg_{window}m'
            if loyalty_col in df_result.columns:
                health_components.append(loyalty_col)

            # Customer stability component
            stability_col = f'customer_stability_{window}m'
            if stability_col in df_result.columns:
                health_components.append(stability_col)

            # Sales volatility component (inverse - lower volatility is better)
            volatility_col = f'RC_M1_SAA_cv_{window}m'
            if volatility_col in df_result.columns:
                # Normalize and invert
                inv_volatility = 1 / (1 + df_result[volatility_col].fillna(0))
                # Replace inf values with NaN
                inv_volatility = inv_volatility.replace([np.inf, -np.inf], np.nan)
                df_result[f'_inv_volatility_{window}m'] = inv_volatility
                health_components.append(f'_inv_volatility_{window}m')

            if len(health_components) > 0:
                # Normalize each component to 0-1 range and take average
                normalized_components = []
                for comp in health_components:
                    min_val = df_result[comp].min()
                    max_val = df_result[comp].max()
                    if max_val - min_val > 0:
                        normalized = (df_result[comp] - min_val) / (max_val - min_val)
                        normalized_components.append(normalized)

                if len(normalized_components) > 0:
                    df_result[f'health_index_{window}m'] = sum(normalized_components) / len(normalized_components)
                    features_created += 1

                    # Clean up temporary columns
                    if f'_inv_volatility_{window}m' in df_result.columns:
                        df_result.drop(columns=[f'_inv_volatility_{window}m'], inplace=True)

        # Risk Index (3, 6, 12 month windows)
        for window in [3, 6, 12]:
            risk_components = []

            # Sales volatility component (high volatility = high risk)
            volatility_col = f'RC_M1_SAA_cv_{window}m'
            if volatility_col in df_result.columns:
                risk_components.append(volatility_col)

            # Negative trend component (declining sales = high risk)
            sales_trend_col = f'RC_M1_SAA_trend_{window}m'
            if sales_trend_col in df_result.columns:
                # Invert negative trends to positive risk values
                df_result[f'_neg_trend_{window}m'] = -df_result[sales_trend_col].fillna(0)
                df_result.loc[df_result[f'_neg_trend_{window}m'] < 0, f'_neg_trend_{window}m'] = 0
                risk_components.append(f'_neg_trend_{window}m')

            # Ranking decline component (worsening rank = high risk)
            rank_change_col = f'RC_M1_SAA_rank_change_{min(window, 6)}m'  # Use 6m max
            if rank_change_col in df_result.columns:
                # Positive rank change = worse rank = high risk
                df_result[f'_rank_decline_{window}m'] = df_result[rank_change_col].fillna(0)
                df_result.loc[df_result[f'_rank_decline_{window}m'] < 0, f'_rank_decline_{window}m'] = 0
                risk_components.append(f'_rank_decline_{window}m')

            # Customer instability component (low stability = high risk)
            stability_col = f'customer_stability_{window}m'
            if stability_col in df_result.columns:
                # Invert stability to get instability
                df_result[f'_instability_{window}m'] = 1 - df_result[stability_col].fillna(0.5)
                risk_components.append(f'_instability_{window}m')

            if len(risk_components) > 0:
                # Normalize each component to 0-1 range and take average
                normalized_components = []
                for comp in risk_components:
                    min_val = df_result[comp].min()
                    max_val = df_result[comp].max()
                    if max_val - min_val > 0:
                        normalized = (df_result[comp] - min_val) / (max_val - min_val)
                        normalized_components.append(normalized)

                if len(normalized_components) > 0:
                    df_result[f'risk_index_{window}m'] = sum(normalized_components) / len(normalized_components)
                    features_created += 1

                    # Clean up temporary columns
                    temp_cols = [f'_neg_trend_{window}m', f'_rank_decline_{window}m', f'_instability_{window}m']
                    for temp_col in temp_cols:
                        if temp_col in df_result.columns:
                            df_result.drop(columns=[temp_col], inplace=True)

        # Growth Index (3, 6, 12 month windows)
        for window in [3, 6, 12]:
            growth_components = []

            # Sales change component
            sales_change_col = f'RC_M1_SAA_change_{min(window, 12)}m'  # Use 12m max
            if sales_change_col in df_result.columns:
                growth_components.append(sales_change_col)

            # Customer change component
            customer_change_col = f'RC_M1_UE_CUS_CN_change_{min(window, 12)}m'
            if customer_change_col in df_result.columns:
                growth_components.append(customer_change_col)

            # Sales trend component
            sales_trend_col = f'RC_M1_SAA_trend_{window}m'
            if sales_trend_col in df_result.columns:
                growth_components.append(sales_trend_col)

            # New customer trend component
            new_customer_trend_col = f'customer_new_trend_{window}m'
            if new_customer_trend_col in df_result.columns:
                growth_components.append(new_customer_trend_col)

            if len(growth_components) > 0:
                # Normalize each component to -1 to 1 range and take average
                normalized_components = []
                for comp in growth_components:
                    min_val = df_result[comp].min()
                    max_val = df_result[comp].max()
                    if max_val - min_val > 0:
                        # Normalize to -1 to 1 (preserving negative values)
                        abs_max = max(abs(min_val), abs(max_val))
                        if abs_max > 0:
                            normalized = df_result[comp] / abs_max
                            normalized_components.append(normalized)

                if len(normalized_components) > 0:
                    df_result[f'growth_index_{window}m'] = sum(normalized_components) / len(normalized_components)
                    features_created += 1

        print(f"Created {features_created} composite indicator features")

        return df_result

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Create interaction features by multiplying pairs of features.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs to interact

        Returns:
            DataFrame with interaction features added
        """
        print(f"\nCreating interaction features...")
        print(f"Feature pairs: {len(feature_pairs)}")

        df_result = df.copy()
        features_created = 0

        for feat1, feat2 in feature_pairs:
            if feat1 in df_result.columns and feat2 in df_result.columns:
                interaction_col_name = f"{feat1}_X_{feat2}"
                df_result[interaction_col_name] = df_result[feat1] * df_result[feat2]
                features_created += 1
            else:
                print(f"Warning: One or both features not found: {feat1}, {feat2}")

        print(f"Created {features_created} interaction features")

        return df_result

    def create_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create ratio features by dividing numerator by denominator.

        Args:
            df: Input DataFrame
            numerator_cols: List of numerator columns
            denominator_cols: List of denominator columns

        Returns:
            DataFrame with ratio features added
        """
        print(f"\nCreating ratio features...")
        print(f"Numerators: {len(numerator_cols)}")
        print(f"Denominators: {len(denominator_cols)}")

        df_result = df.copy()
        features_created = 0

        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col in df_result.columns and den_col in df_result.columns:
                    ratio_col_name = f"{num_col}_div_{den_col}"
                    ratio_values = df_result[num_col] / df_result[den_col].replace(0, np.nan)
                    # Replace inf values with NaN
                    ratio_values = ratio_values.replace([np.inf, -np.inf], np.nan)
                    df_result[ratio_col_name] = ratio_values
                    features_created += 1
                else:
                    print(f"Warning: One or both features not found: {num_col}, {den_col}")

        print(f"Created {features_created} ratio features")

        return df_result
