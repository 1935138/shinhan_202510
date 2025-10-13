"""Feature Engineering Module

This module contains feature engineering classes for creating time series,
customer behavior, composite features, and interval pattern features.
"""

from .time_series import TimeSeriesFeatureEngine
from .customer import CustomerFeatureEngine
from .composite import CompositeFeatureEngine
from .interval_patterns import IntervalPatternFeatureEngine

__all__ = [
    'TimeSeriesFeatureEngine',
    'CustomerFeatureEngine',
    'CompositeFeatureEngine',
    'IntervalPatternFeatureEngine',
]
