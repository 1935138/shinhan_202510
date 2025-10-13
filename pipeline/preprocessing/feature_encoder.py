"""
Feature Encoder Module

구간 데이터 인코딩 및 타겟 변수 생성 기능 제공
- 6단계 구간 변수 → 숫자 변환
- 타겟 변수 생성 (is_closed, will_close_1m, will_close_3m)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class FeatureEncoder:
    """특성 인코딩 및 타겟 변수 생성을 위한 클래스"""

    def __init__(self):
        """구간 매핑 초기화"""
        # 6단계 구간 → 숫자 매핑
        # 실제 데이터에 존재하는 패턴만 포함
        self.interval_mapping = {
            # 퍼센트 형식 (MCT_OPE_MS_CN, RC_M1_* 컬럼)
            '1_10%이하': 1,
            '2_10-25%': 2,
            '3_25-50%': 3,
            '4_50-75%': 4,
            '5_75-90%': 5,
            '6_90%초과(하위 10% 이하)': 6,
            # 한글 구간 형식 (APV_CE_RAT 컬럼)
            '1_상위1구간': 1,
            '2_상위2구간': 2,
            '3_상위3구간': 3,
            '4_상위4구간': 4,
            '5_상위5구간': 5,
            '6_상위6구간(하위1구간)': 6,
        }

        # 역매핑 (디버깅용)
        self.reverse_mapping = {
            1: '1_10%이하',
            2: '2_10-25%',
            3: '3_25-50%',
            4: '4_50-75%',
            5: '5_75-90%',
            6: '6_90%초과(하위 10% 이하)'
        }

    def identify_interval_columns(self, df: pd.DataFrame) -> List[str]:
        """
        구간 변수 컬럼 식별

        구간 변수는 '_PCE_RT' 또는 '_RAT' 등으로 끝나는 경우가 많음

        Args:
            df: 데이터프레임

        Returns:
            구간 변수 컬럼 리스트
        """
        interval_cols = []

        for col in df.columns:
            # object 타입 중에서 구간 패턴을 포함하는 컬럼
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().unique()[:10]
                sample_str = ' '.join(str(v) for v in sample_values)

                # '1_', '2_' 등의 패턴이 있으면 구간 변수로 판단
                if any(f'{i}_' in sample_str for i in range(1, 7)):
                    interval_cols.append(col)

        print(f"\nFound {len(interval_cols)} interval columns:")
        for col in interval_cols[:10]:  # 처음 10개만 출력
            print(f"  - {col}")
        if len(interval_cols) > 10:
            print(f"  ... and {len(interval_cols) - 10} more")

        return interval_cols

    def encode_interval_column(
        self,
        df: pd.DataFrame,
        column: str,
        keep_original: bool = False
    ) -> pd.DataFrame:
        """
        단일 구간 변수 인코딩

        Args:
            df: 데이터프레임
            column: 인코딩할 컬럼명
            keep_original: 원본 컬럼 유지 여부

        Returns:
            인코딩된 데이터프레임
        """
        df_encoded = df.copy()

        if column not in df.columns:
            print(f"Warning: Column '{column}' not found")
            return df_encoded

        # 원본 백업 (옵션)
        if keep_original:
            df_encoded[f'{column}_original'] = df_encoded[column]

        # 매핑 적용
        df_encoded[column] = df_encoded[column].map(self.interval_mapping)

        # 매핑되지 않은 값 확인
        unmapped = df[column][df_encoded[column].isnull() & df[column].notna()]
        if len(unmapped) > 0:
            unique_unmapped = unmapped.unique()
            print(f"Warning: {len(unmapped)} unmapped values in '{column}':")
            print(f"  Unique values: {unique_unmapped[:5]}")

        return df_encoded

    def encode_all_interval_columns(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        keep_original: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        모든 구간 변수 인코딩

        Args:
            df: 데이터프레임
            columns: 인코딩할 컬럼 리스트 (None이면 자동 식별)
            keep_original: 원본 컬럼 유지 여부

        Returns:
            (인코딩된 데이터프레임, 인코딩된 컬럼 리스트)
        """
        print("\n" + "="*60)
        print("ENCODING INTERVAL COLUMNS")
        print("="*60)

        df_encoded = df.copy()

        # 컬럼 자동 식별
        if columns is None:
            columns = self.identify_interval_columns(df)

        print(f"\nEncoding {len(columns)} columns...")

        encoded_cols = []
        for col in columns:
            df_encoded = self.encode_interval_column(df_encoded, col, keep_original)
            encoded_cols.append(col)

        print(f"\nSuccessfully encoded {len(encoded_cols)} columns")

        return df_encoded, encoded_cols

    def create_target_variables(
        self,
        df: pd.DataFrame,
        close_date_col: str = 'MCT_ME_D',
        date_col: str = 'TA_YM',
        merchant_col: str = 'ENCODED_MCT',
        prediction_window: int = 3
    ) -> pd.DataFrame:
        """
        타겟 변수 생성 (Data Leakage 방지)

        생성 변수:
        - will_close_1m: 1개월 내 폐업 예정 (0/1) - 미래 1개월 예측
        - will_close_3m: 3개월 내 폐업 예정 (0/1) - 미래 3개월 예측 (주 타겟)
        - months_until_close: 폐업까지 남은 개월 수 (참고용, feature로 사용 금지)
        - is_valid_for_training: 학습에 사용 가능한 데이터인지 여부 (폐업 직전 제외)

        주의:
        - is_closed 변수는 data leakage를 유발하므로 생성하지 않음
        - months_until_close <= 0인 데이터는 학습에서 제외해야 함 (이미 폐업했거나 폐업 당월)

        Args:
            df: 데이터프레임
            close_date_col: 폐업일 컬럼명
            date_col: 년월 컬럼명
            merchant_col: 가맹점 ID 컬럼명
            prediction_window: 예측 윈도우 (개월)

        Returns:
            타겟 변수가 추가된 데이터프레임
        """
        print("\n" + "="*60)
        print("CREATING TARGET VARIABLES (NO DATA LEAKAGE)")
        print("="*60)

        df_target = df.copy()

        # 1. 날짜 변환
        # TA_YM: YYYYMM 형식 → datetime
        df_target['date_parsed'] = pd.to_datetime(
            df_target[date_col].astype(str),
            format='%Y%m',
            errors='coerce'
        )

        # MCT_ME_D: 폐업일 → datetime (YYYYMMDD 형식)
        # float/int → str → datetime 변환
        has_close_date = df_target[close_date_col].notna()
        df_target['close_date_parsed'] = pd.to_datetime(
            df_target[close_date_col].astype('Int64').astype(str),
            format='%Y%m%d',
            errors='coerce'
        )

        # 2. 폐업까지 남은 개월 수 계산
        df_target['months_until_close'] = np.nan

        # 폐업 가맹점만 계산
        df_target.loc[has_close_date, 'months_until_close'] = (
            (df_target.loc[has_close_date, 'close_date_parsed'].dt.year -
             df_target.loc[has_close_date, 'date_parsed'].dt.year) * 12 +
            (df_target.loc[has_close_date, 'close_date_parsed'].dt.month -
             df_target.loc[has_close_date, 'date_parsed'].dt.month)
        )

        # 3. 미래 N개월 내 폐업 예정 (올바른 방법)
        # months_until_close > 0: 아직 폐업하지 않은 시점
        # months_until_close <= N: N개월 이내에 폐업 예정
        df_target['will_close_1m'] = (
            (df_target['months_until_close'] > 0) &
            (df_target['months_until_close'] <= 1)
        ).astype(int)

        df_target['will_close_3m'] = (
            (df_target['months_until_close'] > 0) &
            (df_target['months_until_close'] <= 3)
        ).astype(int)

        # 4. 학습 데이터로 사용 가능한지 여부
        # - 영업 중인 가맹점 (close_date가 없음): 사용 가능
        # - 폐업 가맹점이지만 아직 폐업 전 (months_until_close > 0): 사용 가능
        # - 이미 폐업했거나 폐업 당월 (months_until_close <= 0): 사용 불가 (data leakage)
        df_target['is_valid_for_training'] = (
            (~has_close_date) |  # 영업 중
            (df_target['months_until_close'] > 0)  # 폐업 전
        ).astype(int)

        # 통계 출력
        print("\n" + "-"*60)
        print("TARGET VARIABLE STATISTICS")
        print("-"*60)

        total_count = len(df_target)
        valid_count = df_target['is_valid_for_training'].sum()
        invalid_count = total_count - valid_count

        print(f"Total records: {total_count:,}")
        print(f"Valid for training: {valid_count:,} ({valid_count/total_count*100:.2f}%)")
        print(f"Invalid (already closed): {invalid_count:,} ({invalid_count/total_count*100:.2f}%)")

        print(f"\nTarget distribution (valid data only):")
        valid_data = df_target[df_target['is_valid_for_training'] == 1]
        print(f"will_close_1m = 1: {valid_data['will_close_1m'].sum():,} ({valid_data['will_close_1m'].sum()/len(valid_data)*100:.2f}%)")
        print(f"will_close_3m = 1: {valid_data['will_close_3m'].sum():,} ({valid_data['will_close_3m'].sum()/len(valid_data)*100:.2f}%)")

        if df_target['months_until_close'].notna().sum() > 0:
            print(f"\nMonths until close (for merchants with close date):")
            print(df_target[df_target['months_until_close'].notna()]['months_until_close'].describe())

        # 임시 컬럼 제거
        df_target.drop(['date_parsed', 'close_date_parsed'], axis=1, inplace=True)

        print("\n" + "="*60)
        print("IMPORTANT: Use only records where is_valid_for_training=1")
        print("="*60)

        return df_target

    def get_encoding_summary(self, df: pd.DataFrame, encoded_cols: List[str]) -> pd.DataFrame:
        """
        인코딩 결과 요약

        Args:
            df: 인코딩된 데이터프레임
            encoded_cols: 인코딩된 컬럼 리스트

        Returns:
            요약 데이터프레임
        """
        summary_data = []

        for col in encoded_cols:
            if col in df.columns:
                value_counts = df[col].value_counts().sort_index()
                summary_data.append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum(),
                    'distribution': dict(value_counts)
                })

        return pd.DataFrame(summary_data)


class DateEncoder:
    """날짜 관련 특성 인코딩"""

    @staticmethod
    def extract_date_features(
        df: pd.DataFrame,
        date_col: str = 'TA_YM'
    ) -> pd.DataFrame:
        """
        날짜로부터 특성 추출

        추출 특성:
        - year: 연도
        - month: 월
        - quarter: 분기
        - month_sin, month_cos: 월의 주기성 (순환 인코딩)

        Args:
            df: 데이터프레임
            date_col: 날짜 컬럼 (YYYYMM 형식)

        Returns:
            날짜 특성이 추가된 데이터프레임
        """
        print(f"\nExtracting date features from '{date_col}'...")

        df_date = df.copy()

        # YYYYMM → datetime
        df_date['date_temp'] = pd.to_datetime(
            df_date[date_col].astype(str),
            format='%Y%m',
            errors='coerce'
        )

        # 특성 추출
        df_date['year'] = df_date['date_temp'].dt.year
        df_date['month'] = df_date['date_temp'].dt.month
        df_date['quarter'] = df_date['date_temp'].dt.quarter

        # 주기성 인코딩 (월: 1~12 → sin/cos)
        df_date['month_sin'] = np.sin(2 * np.pi * df_date['month'] / 12)
        df_date['month_cos'] = np.cos(2 * np.pi * df_date['month'] / 12)

        # 임시 컬럼 제거
        df_date.drop('date_temp', axis=1, inplace=True)

        print(f"Created date features: year, month, quarter, month_sin, month_cos")

        return df_date


def encode_features_and_targets(
    df: pd.DataFrame,
    encode_intervals: bool = True,
    create_targets: bool = True,
    extract_dates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    편의 함수: 모든 인코딩을 한번에 수행

    Args:
        df: 데이터프레임
        encode_intervals: 구간 변수 인코딩 여부
        create_targets: 타겟 변수 생성 여부
        extract_dates: 날짜 특성 추출 여부

    Returns:
        (인코딩된 데이터프레임, 생성된 컬럼 정보)
    """
    df_encoded = df.copy()
    created_cols = {
        'interval_encoded': [],
        'target_vars': [],
        'date_features': []
    }

    # 1. 구간 변수 인코딩
    if encode_intervals:
        encoder = FeatureEncoder()
        df_encoded, interval_cols = encoder.encode_all_interval_columns(df_encoded)
        created_cols['interval_encoded'] = interval_cols

    # 2. 타겟 변수 생성
    if create_targets:
        encoder = FeatureEncoder()
        df_encoded = encoder.create_target_variables(df_encoded)
        created_cols['target_vars'] = [
            'will_close_1m',
            'will_close_3m',
            'months_until_close',
            'is_valid_for_training'
        ]

    # 3. 날짜 특성 추출
    if extract_dates:
        df_encoded = DateEncoder.extract_date_features(df_encoded)
        created_cols['date_features'] = [
            'year',
            'month',
            'quarter',
            'month_sin',
            'month_cos'
        ]

    print("\n" + "="*60)
    print("ENCODING SUMMARY")
    print("="*60)
    print(f"Interval columns encoded: {len(created_cols['interval_encoded'])}")
    print(f"Target variables created: {len(created_cols['target_vars'])}")
    print(f"Date features created: {len(created_cols['date_features'])}")

    return df_encoded, created_cols


if __name__ == "__main__":
    # 테스트 실행
    print("Testing FeatureEncoder...")

    # 샘플 데이터 생성
    test_data = {
        'ENCODED_MCT': [1, 1, 2, 2, 3],
        'TA_YM': [202401, 202402, 202401, 202402, 202401],
        'M1_SME_RY_SAA_RAT': ['3_25-50%', '4_50-75%', '1_10%이하', '2_10-25%', '5_75-90%'],
        'M12_SME_RY_SAA_PCE_RT': ['2_10-25%', '3_25-50%', '1_10%이하', '1_10%이하', '6_90%초과'],
        'MCT_ME_D': [None, None, '2024-03-15', '2024-03-15', None]
    }
    df_test = pd.DataFrame(test_data)

    print("\nOriginal data:")
    print(df_test)

    # 인코딩 수행
    df_encoded, created = encode_features_and_targets(df_test)

    print("\nEncoded data (first 10 columns):")
    print(df_encoded.iloc[:, :10])

    print("\nTarget variables:")
    if 'will_close_1m' in df_encoded.columns:
        print(df_encoded[['will_close_1m', 'will_close_3m', 'is_valid_for_training']])
