"""
Preprocessing Module

데이터 전처리 관련 기능:
- data_loader: 데이터 로드 및 병합
- missing_handler: 결측값 처리
- feature_encoder: 구간 인코딩 및 타겟 변수 생성
"""

from .data_loader import DataLoader, load_and_merge_data
from .missing_handler import MissingValueHandler, process_missing_values
from .feature_encoder import (
    FeatureEncoder,
    DateEncoder,
    encode_features_and_targets
)

__all__ = [
    'DataLoader',
    'load_and_merge_data',
    'MissingValueHandler',
    'process_missing_values',
    'FeatureEncoder',
    'DateEncoder',
    'encode_features_and_targets'
]
