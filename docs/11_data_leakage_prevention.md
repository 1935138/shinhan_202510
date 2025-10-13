# Data Leakage 방지 가이드

## 개요

이 문서는 가맹점 폐업 예측 모델에서 **data leakage**를 방지하기 위한 지침을 설명합니다.

## 문제점: `is_closed` 변수의 Data Leakage

### 기존 문제

이전 버전에서 `is_closed` 변수는 다음과 같이 정의되었습니다:

```python
# ❌ 잘못된 방법 - DATA LEAKAGE!
df['is_closed'] = df['MCT_ME_D'].notna().astype(int)
```

**문제점:**
- `is_closed=1`이면 "이 가맹점은 미래에 폐업할 것이다"라는 정보를 포함
- 폐업 가맹점의 **모든 과거 레코드**에 `is_closed=1`이 표시됨
- 모델이 미래 정보를 알고 학습하는 것과 같음

**예시:**
```
가맹점 A가 202406에 폐업:
  202301: is_closed=1  ← 미래 정보! (5개월 후 폐업할 것을 이미 알고 있음)
  202302: is_closed=1  ← 미래 정보!
  202303: is_closed=1  ← 미래 정보!
  202304: is_closed=1  ← 미래 정보!
  202305: is_closed=1  ← 미래 정보!
  202406: is_closed=1  ← 폐업 시점
```

이렇게 학습하면 모델이 **100% 정확도**를 보일 수 있지만, 실제 운영에서는 전혀 작동하지 않습니다.

## 해결 방안

### 1. 올바른 타겟 변수 정의

`is_closed` 변수를 제거하고, **미래 예측** 타겟만 사용합니다:

```python
# ✅ 올바른 방법 - NO DATA LEAKAGE
df['will_close_3m'] = (
    (df['months_until_close'] > 0) &  # 아직 폐업 전
    (df['months_until_close'] <= 3)   # 3개월 이내 폐업 예정
).astype(int)
```

**현재 타겟 변수:**
- `will_close_1m`: 1개월 이내 폐업 예정 (0/1)
- `will_close_3m`: 3개월 이내 폐업 예정 (0/1) - **주 타겟**
- `months_until_close`: 폐업까지 남은 개월 수 (참고용, feature로 사용 금지)
- `is_valid_for_training`: 학습 데이터로 사용 가능 여부 (0/1)

### 2. 학습 데이터 필터링

폐업 직전/직후 데이터는 제외해야 합니다:

```python
# ✅ 올바른 필터링
df_train = df[df['is_valid_for_training'] == 1].copy()

# is_valid_for_training == 1 조건:
# - 영업 중인 가맹점 (MCT_ME_D가 NaN)
# - 폐업 예정이지만 아직 폐업 전 (months_until_close > 0)

# is_valid_for_training == 0 조건 (제외):
# - 이미 폐업함 (months_until_close <= 0)
# - 폐업 당월 데이터 (months_until_close == 0)
```

**예시:**
```
가맹점 A가 202406에 폐업:
  202301: is_valid_for_training=1, will_close_3m=0  ← 사용 가능 (5개월 후 폐업)
  202302: is_valid_for_training=1, will_close_3m=0  ← 사용 가능 (4개월 후 폐업)
  202303: is_valid_for_training=1, will_close_3m=1  ← 사용 가능 (3개월 후 폐업, 타겟=1)
  202304: is_valid_for_training=1, will_close_3m=1  ← 사용 가능 (2개월 후 폐업, 타겟=1)
  202305: is_valid_for_training=1, will_close_3m=1  ← 사용 가능 (1개월 후 폐업, 타겟=1)
  202406: is_valid_for_training=0, will_close_3m=0  ← 제외 (폐업 당월)
  202407: (데이터 없음, 이미 폐업)
```

### 3. Feature Engineering 시 주의사항

Feature 생성 시 **절대 미래 정보**를 사용하지 말 것:

```python
# ❌ 잘못된 예 - 미래 데이터 포함
df['sales_ma_3m'] = df['sales'].rolling(3).mean()

# ✅ 올바른 예 - 과거 데이터만 사용
df['sales_ma_3m'] = df.groupby('ENCODED_MCT')['sales'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
```

**항상 `.shift()`를 사용하여 lag를 추가하세요.**

### 4. 제외해야 할 변수

다음 변수들은 **절대 feature로 사용하면 안 됩니다:**

```python
exclude_cols = [
    'ENCODED_MCT',           # 가맹점 ID
    'TA_YM',                 # 날짜 (year, month 등 파생 변수는 사용 가능)
    'MCT_ME_D',              # 폐업일 (미래 정보)
    'will_close_1m',         # 타겟 변수
    'will_close_3m',         # 타겟 변수
    'months_until_close',    # 미래 정보 (폐업까지 남은 개월 수)
    'is_valid_for_training', # 학습 필터용
]
```

## 코드 체크리스트

### 1. 데이터 로드 시

```python
from pipeline.preprocessing import load_and_merge_data, FeatureEncoder

# 데이터 로드 및 타겟 생성
df = load_and_merge_data()
encoder = FeatureEncoder()
df = encoder.create_target_variables(df)

# ✅ 항상 is_valid_for_training=1로 필터링
df_valid = df[df['is_valid_for_training'] == 1].copy()

print(f"Total records: {len(df):,}")
print(f"Valid for training: {len(df_valid):,}")
```

### 2. Feature Engineering 시

```python
# ✅ Feature 목록에서 타겟 변수 제외
exclude_cols = [
    'ENCODED_MCT', 'TA_YM', 'MCT_ME_D',
    'will_close_1m', 'will_close_3m',
    'months_until_close', 'is_valid_for_training'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
```

### 3. 모델 학습 시

```python
# ✅ 올바른 train/test split
# 시간 기반 분할 사용
train_mask = df_valid['TA_YM'] < 202410
test_mask = df_valid['TA_YM'] >= 202410

df_train = df_valid[train_mask]
df_test = df_valid[test_mask]

# 타겟 분리
X_train = df_train[feature_cols]
y_train = df_train['will_close_3m']

X_test = df_test[feature_cols]
y_test = df_test['will_close_3m']
```

### 4. 모델 평가 시

```python
from pipeline.evaluation.metrics import calculate_pr_auc

# ✅ Precision-Recall AUC 사용 (imbalanced data에 적합)
pr_auc = calculate_pr_auc(y_test, y_pred_proba)
print(f"PR-AUC: {pr_auc:.4f}")

# ROC-AUC는 참고용으로만
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f} (참고용)")
```

## 검증 방법

### 1. Target Distribution 확인

```python
print("Target distribution:")
print(df_valid['will_close_3m'].value_counts())
print(f"Positive rate: {df_valid['will_close_3m'].mean()*100:.2f}%")

# 예상 결과: 1~3% 정도의 positive rate (폐업률)
```

### 2. Feature Importance 분석

```python
import shap

# ✅ months_until_close나 MCT_ME_D가 상위에 없어야 함
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type='bar')

# 상위 feature가 매출 추세, 재방문율 등 비즈니스 지표여야 정상
```

### 3. 시간별 성능 확인

```python
# ✅ 시간이 지나도 성능이 일정해야 함
for month in [202410, 202411, 202412]:
    month_mask = df_test['TA_YM'] == month
    if month_mask.sum() > 0:
        X_month = X_test[month_mask]
        y_month = y_test[month_mask]
        y_pred_month = model.predict_proba(X_month)[:, 1]

        pr_auc_month = calculate_pr_auc(y_month, y_pred_month)
        print(f"{month}: PR-AUC = {pr_auc_month:.4f}")

# 만약 첫 달 성능이 매우 높고 이후 급락하면 data leakage 의심
```

## 요약

1. **`is_closed` 변수는 절대 사용하지 않음** - 제거됨
2. **타겟은 `will_close_3m` 사용** - 미래 3개월 예측
3. **`is_valid_for_training=1`로 필터링 필수** - 폐업 직전 데이터 제외
4. **Feature 생성 시 `.shift()` 사용** - 미래 정보 차단
5. **시간 기반 분할** - Train/Test를 날짜로 분리
6. **검증 필수** - Feature importance와 시간별 성능 확인

이 가이드를 따르면 **실제 운영 환경에서도 작동하는** 신뢰할 수 있는 모델을 만들 수 있습니다.
