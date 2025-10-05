"""
Data Loader Module

3개의 데이터셋을 로드하고 병합하는 기능 제공
- Dataset 1: 가맹점 기본정보 (업종, 위치, 폐업일)
- Dataset 2: 월별 매출/이용 현황
- Dataset 3: 월별 고객 정보
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """데이터셋 로드 및 병합을 위한 클래스"""

    def __init__(self):
        """
        Raw 데이터 파일은 프로젝트의 data/raw/ 디렉토리에 고정
        """
        # 프로젝트 루트 디렉토리 (pipeline/preprocessing/data_loader.py 기준)
        project_root = Path(__file__).parent.parent.parent
        raw_data_dir = project_root / "data" / "raw"

        self.dataset1_path = raw_data_dir / "big_data_set1_f.csv"
        self.dataset2_path = raw_data_dir / "big_data_set2_f.csv"
        self.dataset3_path = raw_data_dir / "big_data_set3_f.csv"

    def load_dataset1(self) -> pd.DataFrame:
        """
        Dataset 1 로드: 가맹점 기본정보

        Returns:
            가맹점 기본정보 데이터프레임
        """
        print(f"Loading Dataset 1: {self.dataset1_path}")
        df = pd.read_csv(self.dataset1_path, encoding='cp949')
        print(f"Dataset 1 shape: {df.shape}")
        print(f"Dataset 1 columns: {list(df.columns)}")
        return df

    def load_dataset2(self) -> pd.DataFrame:
        """
        Dataset 2 로드: 월별 매출/이용 현황

        Returns:
            월별 매출/이용 데이터프레임
        """
        print(f"Loading Dataset 2: {self.dataset2_path}")
        df = pd.read_csv(self.dataset2_path, encoding='cp949')
        print(f"Dataset 2 shape: {df.shape}")
        print(f"Dataset 2 columns: {list(df.columns)[:10]}...")  # 처음 10개만 출력
        return df

    def load_dataset3(self) -> pd.DataFrame:
        """
        Dataset 3 로드: 월별 고객 정보

        Returns:
            월별 고객 정보 데이터프레임
        """
        print(f"Loading Dataset 3: {self.dataset3_path}")
        df = pd.read_csv(self.dataset3_path, encoding='cp949')
        print(f"Dataset 3 shape: {df.shape}")
        print(f"Dataset 3 columns: {list(df.columns)[:10]}...")  # 처음 10개만 출력
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        모든 데이터셋 로드

        Returns:
            (dataset1, dataset2, dataset3) 튜플
        """
        df1 = self.load_dataset1()
        df2 = self.load_dataset2()
        df3 = self.load_dataset3()
        return df1, df2, df3

    def merge_datasets(
        self,
        df1: Optional[pd.DataFrame] = None,
        df2: Optional[pd.DataFrame] = None,
        df3: Optional[pd.DataFrame] = None,
        validate_merge: bool = True
    ) -> pd.DataFrame:
        """
        3개 데이터셋 병합

        병합 전략:
        1. Dataset 2 + Dataset 3: ENCODED_MCT, TA_YM 기준 병합
           → 같은 가맹점의 같은 월 데이터를 옆으로 붙임
        2. 위 결과 + Dataset 1: ENCODED_MCT 기준 병합
           → 가맹점별 고정 정보를 모든 월 데이터에 추가

        Args:
            df1: Dataset 1 (None이면 자동 로드)
            df2: Dataset 2 (None이면 자동 로드)
            df3: Dataset 3 (None이면 자동 로드)
            validate_merge: 병합 결과 검증 여부

        Returns:
            병합된 데이터프레임
        """
        # 데이터가 제공되지 않으면 로드
        if df1 is None or df2 is None or df3 is None:
            print("Loading datasets...")
            df1, df2, df3 = self.load_all()

        print("\n" + "="*60)
        print("STEP 1: Merging Dataset 2 + Dataset 3")
        print("="*60)

        # Dataset 2와 3 병합 (가맹점ID + 년월 기준)
        df_merged = df2.merge(
            df3,
            on=['ENCODED_MCT', 'TA_YM'],
            how='inner',
            suffixes=('_sales', '_customer')
        )

        print(f"Dataset 2 shape: {df2.shape}")
        print(f"Dataset 3 shape: {df3.shape}")
        print(f"Merged (2+3) shape: {df_merged.shape}")

        if validate_merge:
            # 병합 손실 확인
            original_rows = len(df2)
            merged_rows = len(df_merged)
            loss_rate = (original_rows - merged_rows) / original_rows * 100
            print(f"Merge loss: {original_rows - merged_rows} rows ({loss_rate:.2f}%)")

        print("\n" + "="*60)
        print("STEP 2: Merging (2+3) + Dataset 1")
        print("="*60)

        # 위 결과와 Dataset 1 병합 (가맹점ID 기준)
        df_full = df_merged.merge(
            df1,
            on='ENCODED_MCT',
            how='left',
            suffixes=('', '_info')
        )

        print(f"Merged (2+3) shape: {df_merged.shape}")
        print(f"Dataset 1 shape: {df1.shape}")
        print(f"Final merged shape: {df_full.shape}")

        if validate_merge:
            # 가맹점 정보 병합 확인
            null_counts = df_full.isnull().sum()
            if null_counts.sum() > 0:
                print(f"\nWarning: Found {null_counts.sum()} null values after merge")
                print("Top 5 columns with nulls:")
                print(null_counts[null_counts > 0].head())

        print("\n" + "="*60)
        print("MERGE SUMMARY")
        print("="*60)
        print(f"Total rows: {len(df_full):,}")
        print(f"Total columns: {len(df_full.columns)}")
        print(f"Unique merchants: {df_full['ENCODED_MCT'].nunique():,}")
        print(f"Date range: {df_full['TA_YM'].min()} ~ {df_full['TA_YM'].max()}")

        return df_full

    def get_merge_info(self, df: pd.DataFrame) -> dict:
        """
        병합된 데이터의 정보 요약

        Args:
            df: 병합된 데이터프레임

        Returns:
            정보 딕셔너리
        """
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_merchants': df['ENCODED_MCT'].nunique(),
            'date_range': (df['TA_YM'].min(), df['TA_YM'].max()),
            'months_count': df['TA_YM'].nunique(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        return info


def load_and_merge_data() -> pd.DataFrame:
    """
    편의 함수: 데이터 로드 및 병합을 한번에 수행

    Returns:
        병합된 데이터프레임
    """
    loader = DataLoader()
    df_merged = loader.merge_datasets()
    return df_merged


if __name__ == "__main__":
    # 테스트 실행
    print("Testing DataLoader...")
    loader = DataLoader()
    df = loader.merge_datasets()

    info = loader.get_merge_info(df)
    print("\n" + "="*60)
    print("FINAL INFO")
    print("="*60)
    for key, value in info.items():
        print(f"{key}: {value}")
