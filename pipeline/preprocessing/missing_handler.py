"""
Missing Value Handler Module

결측값(SV: -999999.9) 처리 기능 제공
- SV 감지 및 변환
- 배달 관련 변수: 0으로 대체 (배달 미운영)
- 고객 정보: 중앙값 또는 플래그 생성
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class MissingValueHandler:
    """결측값 처리를 위한 클래스"""

    def __init__(self, sv_value: float = -999999.9):
        """
        Args:
            sv_value: Special Value (결측값 표시) 기본값
        """
        self.sv_value = sv_value
        self.imputation_stats = {}  # 대체 통계 저장

    def detect_sv(self, df: pd.DataFrame, threshold: float = -999999.0) -> pd.DataFrame:
        """
        SV 값을 감지하고 보고

        Args:
            df: 데이터프레임
            threshold: SV 판단 기준값

        Returns:
            SV 통계 데이터프레임
        """
        print("Detecting SV (Special Values)...")

        sv_stats = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                sv_count = (df[col] <= threshold).sum()
                if sv_count > 0:
                    sv_ratio = sv_count / len(df) * 100
                    sv_stats.append({
                        'column': col,
                        'sv_count': sv_count,
                        'sv_ratio': sv_ratio,
                        'total_rows': len(df)
                    })

        sv_df = pd.DataFrame(sv_stats).sort_values('sv_count', ascending=False)

        if len(sv_df) > 0:
            print(f"\nFound SV in {len(sv_df)} columns")
            print(f"Total SV values: {sv_df['sv_count'].sum():,}")
            print("\nTop 10 columns with SV:")
            print(sv_df.head(10).to_string(index=False))
        else:
            print("No SV values found")

        return sv_df

    def replace_sv_with_nan(self, df: pd.DataFrame, threshold: float = -999999.0) -> pd.DataFrame:
        """
        SV 값을 NaN으로 변환

        Args:
            df: 데이터프레임
            threshold: SV 판단 기준값

        Returns:
            SV가 NaN으로 변환된 데이터프레임
        """
        print(f"\nReplacing SV (values <= {threshold}) with NaN...")

        df_clean = df.copy()
        replaced_count = 0

        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                sv_mask = df_clean[col] <= threshold
                count = sv_mask.sum()
                if count > 0:
                    df_clean.loc[sv_mask, col] = np.nan
                    replaced_count += count

        print(f"Replaced {replaced_count:,} SV values with NaN")
        return df_clean

    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        컬럼을 카테고리별로 분류

        Args:
            df: 데이터프레임

        Returns:
            카테고리별 컬럼 리스트 딕셔너리
        """
        column_types = {
            'delivery': [],      # 배달 관련
            'sales': [],         # 매출 관련
            'customer': [],      # 고객 관련
            'ratio': [],         # 비율/순위 관련
            'category': [],      # 범주형
            'date': [],          # 날짜
            'id': [],            # ID
            'other': []          # 기타
        }

        for col in df.columns:
            col_lower = col.lower()

            # 배달 관련
            if 'dlv' in col_lower or 'delivery' in col_lower or '배달' in col_lower:
                column_types['delivery'].append(col)
            # 매출 관련
            elif 'saa' in col_lower or 'sales' in col_lower or '매출' in col_lower:
                column_types['sales'].append(col)
            # 고객 관련
            elif 'cus' in col_lower or 'customer' in col_lower or '고객' in col_lower:
                column_types['customer'].append(col)
            # 비율/순위
            elif 'rat' in col_lower or 'pce' in col_lower or 'ratio' in col_lower or 'rank' in col_lower:
                column_types['ratio'].append(col)
            # 날짜
            elif 'ym' in col_lower or 'date' in col_lower or '_d' in col_lower:
                column_types['date'].append(col)
            # ID
            elif 'mct' in col_lower or 'id' in col_lower or 'encoded' in col_lower:
                column_types['id'].append(col)
            # 범주형 (object 타입)
            elif df[col].dtype == 'object':
                column_types['category'].append(col)
            # 기타
            else:
                column_types['other'].append(col)

        # 결과 출력
        print("\n" + "="*60)
        print("COLUMN TYPE CLASSIFICATION")
        print("="*60)
        for cat, cols in column_types.items():
            if cols:
                print(f"{cat.upper()}: {len(cols)} columns")

        return column_types

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'auto',
        column_types: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        결측값 대체

        전략:
        - 배달 관련: 0으로 대체 (배달 미운영)
        - 매출 관련: 중앙값
        - 고객 관련: 중앙값
        - 비율 관련: 중앙값
        - 범주형: 최빈값

        Args:
            df: 데이터프레임
            strategy: 대체 전략 ('auto', 'median', 'mean', 'zero')
            column_types: 컬럼 타입 분류 (None이면 자동 분류)

        Returns:
            결측값이 대체된 데이터프레임
        """
        print("\n" + "="*60)
        print("IMPUTING MISSING VALUES")
        print("="*60)

        df_imputed = df.copy()

        # 컬럼 타입 자동 분류
        if column_types is None:
            column_types = self.identify_column_types(df)

        imputation_log = []

        # 1. 배달 관련: 0으로 대체
        for col in column_types['delivery']:
            if df_imputed[col].isnull().sum() > 0:
                null_count = df_imputed[col].isnull().sum()
                df_imputed[col] = df_imputed[col].fillna(0)
                imputation_log.append({
                    'column': col,
                    'type': 'delivery',
                    'method': 'zero',
                    'null_count': null_count
                })
                self.imputation_stats[col] = {'method': 'zero', 'value': 0}

        # 2. 매출/고객/비율 관련: 중앙값 (숫자형만)
        for col_type in ['sales', 'customer', 'ratio']:
            for col in column_types[col_type]:
                if df_imputed[col].isnull().sum() > 0:
                    # 숫자형만 처리 (구간 변수 제외)
                    if df_imputed[col].dtype in ['float64', 'int64']:
                        null_count = df_imputed[col].isnull().sum()
                        median_val = df_imputed[col].median()
                        df_imputed[col] = df_imputed[col].fillna(median_val)
                        imputation_log.append({
                            'column': col,
                            'type': col_type,
                            'method': 'median',
                            'null_count': null_count
                        })
                        self.imputation_stats[col] = {'method': 'median', 'value': median_val}

        # 3. 범주형: 최빈값 또는 'Unknown'
        for col in column_types['category']:
            if df_imputed[col].isnull().sum() > 0:
                null_count = df_imputed[col].isnull().sum()
                if len(df_imputed[col].mode()) > 0:
                    mode_val = df_imputed[col].mode()[0]
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                    method = 'mode'
                    fill_val = mode_val
                else:
                    df_imputed[col] = df_imputed[col].fillna('Unknown')
                    method = 'constant'
                    fill_val = 'Unknown'

                imputation_log.append({
                    'column': col,
                    'type': 'category',
                    'method': method,
                    'null_count': null_count
                })
                self.imputation_stats[col] = {'method': method, 'value': fill_val}

        # 4. 기타: 중앙값 (숫자형) 또는 'Unknown' (문자형)
        for col in column_types['other']:
            if df_imputed[col].isnull().sum() > 0:
                null_count = df_imputed[col].isnull().sum()
                if df_imputed[col].dtype in ['float64', 'int64']:
                    median_val = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_val)
                    method = 'median'
                    fill_val = median_val
                else:
                    df_imputed[col] = df_imputed[col].fillna('Unknown')
                    method = 'constant'
                    fill_val = 'Unknown'

                imputation_log.append({
                    'column': col,
                    'type': 'other',
                    'method': method,
                    'null_count': null_count
                })
                self.imputation_stats[col] = {'method': method, 'value': fill_val}

        # 로그 출력
        if imputation_log:
            log_df = pd.DataFrame(imputation_log)
            print(f"\nImputed {len(imputation_log)} columns")
            print(f"Total null values filled: {log_df['null_count'].sum():,}")
            print("\nImputation summary by type:")
            print(log_df.groupby('type')['null_count'].sum().to_string())
        else:
            print("\nNo missing values to impute")

        # 최종 결측값 확인
        remaining_nulls = df_imputed.isnull().sum().sum()
        print(f"\nRemaining null values: {remaining_nulls:,}")

        return df_imputed

    def create_missing_flags(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        prefix: str = 'is_missing_'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        결측값 플래그 생성

        Args:
            df: 데이터프레임
            columns: 플래그를 생성할 컬럼 리스트 (None이면 모든 결측값 있는 컬럼)
            prefix: 플래그 컬럼명 접두사

        Returns:
            (플래그가 추가된 데이터프레임, 생성된 플래그 컬럼 리스트)
        """
        df_flagged = df.copy()
        flag_columns = []

        # 컬럼이 지정되지 않으면 결측값이 있는 모든 컬럼
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()

        print(f"\nCreating missing value flags for {len(columns)} columns...")

        for col in columns:
            if col in df.columns:
                flag_col = f"{prefix}{col}"
                df_flagged[flag_col] = df[col].isnull().astype(int)
                flag_columns.append(flag_col)

        print(f"Created {len(flag_columns)} flag columns")
        return df_flagged, flag_columns

    def get_imputation_report(self) -> pd.DataFrame:
        """
        대체 통계 리포트 생성

        Returns:
            대체 통계 데이터프레임
        """
        if not self.imputation_stats:
            print("No imputation stats available")
            return pd.DataFrame()

        report_data = []
        for col, stats in self.imputation_stats.items():
            report_data.append({
                'column': col,
                'method': stats['method'],
                'value': stats['value']
            })

        return pd.DataFrame(report_data)


def process_missing_values(
    df: pd.DataFrame,
    sv_value: float = -999999.9,
    create_flags: bool = False
) -> pd.DataFrame:
    """
    편의 함수: 결측값 처리를 한번에 수행

    Args:
        df: 데이터프레임
        sv_value: Special Value
        create_flags: 결측값 플래그 생성 여부

    Returns:
        처리된 데이터프레임
    """
    handler = MissingValueHandler(sv_value)

    # SV 감지
    sv_stats = handler.detect_sv(df)

    # SV를 NaN으로 변환
    df_clean = handler.replace_sv_with_nan(df)

    # 결측값 플래그 생성 (옵션)
    if create_flags:
        df_clean, flag_cols = handler.create_missing_flags(df_clean)
        print(f"Created {len(flag_cols)} missing flags")

    # 결측값 대체
    df_imputed = handler.impute_missing_values(df_clean)

    # 리포트
    report = handler.get_imputation_report()
    print("\n" + "="*60)
    print("IMPUTATION REPORT")
    print("="*60)
    if not report.empty:
        print(f"Total columns imputed: {len(report)}")

    return df_imputed


if __name__ == "__main__":
    # 테스트 실행
    print("Testing MissingValueHandler...")

    # 샘플 데이터 생성
    test_data = {
        'ENCODED_MCT': [1, 2, 3, 4, 5],
        'RC_M1_SAA': [1000000, -999999.9, 2000000, -999999.9, 3000000],
        'RC_M1_DLV_SAA': [-999999.9, 500000, -999999.9, 700000, -999999.9],
        'RC_M1_UE_CUS_CN': [100, 200, -999999.9, 400, 500],
        'HPSN_MCT_ZCD_NM': ['치킨', '카페', None, '한식', '일식']
    }
    df_test = pd.DataFrame(test_data)

    print("\nOriginal data:")
    print(df_test)

    df_processed = process_missing_values(df_test, create_flags=True)

    print("\nProcessed data:")
    print(df_processed)
