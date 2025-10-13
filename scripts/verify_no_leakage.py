"""
Data Leakage 검증 스크립트

이 스크립트는 전처리된 데이터가 data leakage 없이 올바르게 생성되었는지 확인합니다.
"""

import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def verify_target_variables(df: pd.DataFrame) -> bool:
    """타겟 변수가 올바르게 생성되었는지 확인"""
    print("="*80)
    print("1. 타겟 변수 검증")
    print("="*80)

    # is_closed 변수가 없어야 함
    if 'is_closed' in df.columns:
        print("❌ FAIL: is_closed 변수가 존재합니다 (data leakage!)")
        return False
    else:
        print("✅ PASS: is_closed 변수가 제거되었습니다")

    # 필수 타겟 변수 확인
    required_targets = ['will_close_1m', 'will_close_3m', 'is_valid_for_training']
    for target in required_targets:
        if target not in df.columns:
            print(f"❌ FAIL: {target} 변수가 없습니다")
            return False

    print("✅ PASS: 모든 필수 타겟 변수가 존재합니다")

    # 타겟 변수 분포 확인
    print("\n타겟 변수 분포:")
    print(f"will_close_1m = 1: {df['will_close_1m'].sum():,} ({df['will_close_1m'].mean()*100:.2f}%)")
    print(f"will_close_3m = 1: {df['will_close_3m'].sum():,} ({df['will_close_3m'].mean()*100:.2f}%)")
    print(f"is_valid_for_training = 1: {df['is_valid_for_training'].sum():,} ({df['is_valid_for_training'].mean()*100:.2f}%)")

    return True


def verify_feature_columns(df: pd.DataFrame) -> bool:
    """Feature 컬럼에 금지된 변수가 없는지 확인"""
    print("\n" + "="*80)
    print("2. Feature 컬럼 검증")
    print("="*80)

    # 절대 feature로 사용하면 안 되는 변수들
    forbidden_features = [
        'is_closed',
        'MCT_ME_D',
        'months_until_close'
    ]

    found_forbidden = []
    for col in forbidden_features:
        if col in df.columns:
            found_forbidden.append(col)

    if found_forbidden:
        print(f"⚠️  WARNING: 다음 변수들은 feature로 사용 금지:")
        for col in found_forbidden:
            print(f"   - {col}")
        print("\n   이 변수들은 모델 학습 시 반드시 제외해야 합니다!")
        return True  # Warning이지만 계속 진행
    else:
        print("✅ PASS: 금지된 feature가 없습니다")

    return True


def verify_valid_data_ratio(df: pd.DataFrame) -> bool:
    """Valid 데이터 비율 확인"""
    print("\n" + "="*80)
    print("3. Valid 데이터 비율 검증")
    print("="*80)

    if 'is_valid_for_training' not in df.columns:
        print("❌ FAIL: is_valid_for_training 컬럼이 없습니다")
        return False

    total = len(df)
    valid = df['is_valid_for_training'].sum()
    invalid = total - valid

    print(f"Total records: {total:,}")
    print(f"Valid (is_valid_for_training=1): {valid:,} ({valid/total*100:.2f}%)")
    print(f"Invalid (is_valid_for_training=0): {invalid:,} ({invalid/total*100:.2f}%)")

    # Invalid 데이터가 2-5% 정도면 정상 (폐업 당월 데이터)
    invalid_ratio = invalid / total * 100
    if invalid_ratio > 10:
        print(f"⚠️  WARNING: Invalid 데이터 비율이 높습니다 ({invalid_ratio:.2f}%)")
        print("   예상: 2-5% (폐업 당월 데이터)")
        return True  # Warning이지만 계속 진행
    else:
        print(f"✅ PASS: Invalid 데이터 비율이 정상 범위입니다 ({invalid_ratio:.2f}%)")

    return True


def verify_target_logic(df: pd.DataFrame) -> bool:
    """타겟 변수 로직 검증"""
    print("\n" + "="*80)
    print("4. 타겟 변수 로직 검증")
    print("="*80)

    if 'will_close_3m' not in df.columns or 'months_until_close' not in df.columns:
        print("❌ FAIL: 필수 컬럼이 없습니다")
        return False

    # will_close_3m = 1인 데이터는 months_until_close가 1~3 범위여야 함
    target_positive = df[df['will_close_3m'] == 1]

    if len(target_positive) > 0:
        months_range = target_positive['months_until_close'].describe()
        print(f"\nwill_close_3m=1인 데이터의 months_until_close 분포:")
        print(months_range)

        # 범위 확인
        min_months = target_positive['months_until_close'].min()
        max_months = target_positive['months_until_close'].max()

        if min_months < 1 or max_months > 3:
            print(f"❌ FAIL: will_close_3m=1 데이터의 months_until_close 범위가 잘못되었습니다")
            print(f"   예상: 1~3, 실제: {min_months}~{max_months}")
            return False
        else:
            print(f"✅ PASS: will_close_3m=1 데이터의 범위가 올바릅니다 ({min_months}~{max_months})")
    else:
        print("⚠️  WARNING: will_close_3m=1인 데이터가 없습니다")

    # is_valid_for_training=0인 데이터는 months_until_close <= 0이어야 함
    invalid_data = df[df['is_valid_for_training'] == 0]

    if len(invalid_data) > 0:
        closed_data = invalid_data[invalid_data['months_until_close'].notna()]
        if len(closed_data) > 0:
            max_months_invalid = closed_data['months_until_close'].max()
            if max_months_invalid > 0:
                print(f"❌ FAIL: is_valid_for_training=0 데이터 중 months_until_close > 0인 데이터가 있습니다")
                return False
            else:
                print(f"✅ PASS: is_valid_for_training=0 데이터의 로직이 올바릅니다")

    return True


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("DATA LEAKAGE 검증 스크립트")
    print("="*80)

    # 전처리된 데이터 로드
    data_path = project_root / 'data' / 'processed' / 'preprocessed_data.csv'

    if not data_path.exists():
        print(f"\n❌ ERROR: 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("   먼저 02_preprocessing.ipynb를 실행하여 전처리된 데이터를 생성하세요.")
        sys.exit(1)

    print(f"\n데이터 로드 중: {data_path}")
    df = pd.read_csv(data_path)
    print(f"데이터 shape: {df.shape}")

    # 검증 실행
    tests = [
        ("타겟 변수 검증", verify_target_variables),
        ("Feature 컬럼 검증", verify_feature_columns),
        ("Valid 데이터 비율 검증", verify_valid_data_ratio),
        ("타겟 변수 로직 검증", verify_target_logic),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(df)
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # 최종 결과
    print("\n" + "="*80)
    print("최종 결과")
    print("="*80)

    all_passed = all(result for _, result in results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    if all_passed:
        print("\n🎉 모든 검증을 통과했습니다!")
        print("   데이터가 data leakage 없이 올바르게 생성되었습니다.")
        print("\n⚠️  주의: 모델 학습 시 반드시 is_valid_for_training=1 데이터만 사용하세요!")
        sys.exit(0)
    else:
        print("\n⚠️  일부 검증에 실패했습니다. 위의 메시지를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
