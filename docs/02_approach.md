# 알고리즘 추천 및 모델링 전략

## 목차
1. [문제 정의](#1-문제-정의)
2. [데이터 특성 분석](#2-데이터-특성-분석)
3. [추천 알고리즘](#3-추천-알고리즘)
4. [모델링 전략](#4-모델링-전략)
5. [Feature Engineering](#5-feature-engineering)
6. [평가 전략](#6-평가-전략)
7. [최종 추천](#7-최종-추천)

---

## 1. 문제 정의

### 목표
영세/중소 요식 가맹점의 경영 위기(매출 급락, 폐업 위험)를 미리 감지하는 **AI 조기 경보 시스템** 개발

### 문제 유형
- **이진 분류**: 폐업 여부 예측
- **회귀**: 매출 급락 예측
- **시계열 이상 탐지**: 비정상 패턴 조기 발견
- **다중 분류**: 매출/고객 수 구간 변화 예측

---

## 2. 데이터 특성 분석

### 2.1 데이터 규모
- **총 가맹점 수**: 4,185개
- **분석 기간**: 2023년 01월 ~ 2024년 12월 (24개월)
- **월별 데이터**: 86,590건
- **업종**: 73개 / **상권**: 21개

### 2.2 주요 특성

#### 극심한 클래스 불균형
- 폐업: 127개 (3.03%)
- 운영중: 4,058개 (96.97%)
- **→ 불균형 처리 필수**

#### 시계열 데이터
- 24개월 월별 데이터
- 매출/고객 추세 변화 포착 가능
- **→ 시계열 특성 활용**

#### 구간화된 데이터 (Dataset 2)
- 대부분 6개 구간으로 범주화
  - 10%이하, 10-25%, 25-50%, 50-75%, 75-90%, 90%초과
- 매출금액, 매출건수, 고객수, 객단가 등
- **→ 순서형 범주 처리 필요**

#### 결측값 (Special Value: -999999.9)
- 배달매출 비율: 66.23%
- 상권 해지 가맹점 비중: 24.74%
- 고객 정보: 2.31~8.46%
- **→ 결측값 처리 전략 필요**

#### 다차원 특성
- 매출 지표 (금액, 건수, 객단가)
- 고객 지표 (재방문율, 신규율, 연령/성별)
- 상권/업종 비교 지표
- **→ 복합적 패턴 학습 필요**

---

## 3. 추천 알고리즘

### 3.1 앙상블 기반 모델 ⭐ (최우선 추천)

#### XGBoost / LightGBM / CatBoost

**장점**:
- 불균형 데이터 처리 용이
  - XGBoost: `scale_pos_weight` 파라미터
  - LightGBM: `is_unbalance=True`
  - CatBoost: `auto_class_weights`
- 범주형/구간 데이터 효과적 처리
- 결측값 자동 처리
- Feature importance 제공 → **위기 신호 해석 가능** (평가 기준)
- 높은 예측 성능

**활용 방안**:
- 폐업 예측 (이진 분류)
- 매출 구간 하락 예측 (다중 분류)
- SHAP values로 인사이트 도출
- 업종/상권별 위험 요인 분석

**구현 예시**:
```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# 불균형 비율 계산
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost 모델
model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=500,
    early_stopping_rounds=50
)

# 시계열 교차 검증
tscv = TimeSeriesSplit(n_splits=5)
```

---

### 3.2 시계열 기반 모델

#### LSTM / GRU (Deep Learning)

**장점**:
- 월별 시퀀스 패턴 학습 (24개월)
- 매출/고객 추세 변화 포착
- 장기 의존성 학습

**활용 방안**:
- 다음 달 매출 구간 예측
- 이상 패턴 조기 감지
- 연속적 하락 추세 학습

**구현 예시**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 시퀀스 길이 설정 (예: 6개월)
sequence_length = 6

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
```

#### Prophet / ARIMA

**장점**:
- 계절성/트렌드 분리
- 간단한 해석
- 빠른 구현

**활용 방안**:
- 매출 예측 및 이상치 탐지
- 업종별 계절성 패턴 분석

---

### 3.3 이상 탐지 모델

#### Isolation Forest

**장점**:
- 레이블 없이 위기 징후 탐지
- 정상 패턴에서 벗어난 가맹점 식별
- 해석 가능한 이상치 점수

**활용 방안**:
- 급격한 매출 하락 감지
- 고객 이탈 패턴 발견
- 비지도 학습으로 새로운 위험 신호 탐색

#### AutoEncoder

**장점**:
- 복잡한 패턴 학습
- 재구성 오차로 이상 탐지
- 정상 운영 패턴 학습

**구현 예시**:
```python
from sklearn.ensemble import IsolationForest

# 정상 가맹점 데이터로 학습
normal_data = df[df['MCT_ME_D'].isna()]

iso_forest = IsolationForest(
    contamination=0.05,  # 5% 이상치 예상
    random_state=42
)

# 이상 점수 계산
anomaly_scores = iso_forest.fit_predict(features)
```

---

### 3.4 생존 분석 (Survival Analysis)

#### Cox Proportional Hazards / Random Survival Forest

**장점**:
- 폐업까지의 시간 예측
- 시간에 따른 위험 요인 식별
- 검열 데이터(censored data) 처리 가능

**활용 방안**:
- 운영 기간별 폐업 위험도 계산
- 개설 후 경과 시간 고려
- 위험 비율(Hazard Ratio) 해석

**구현 예시**:
```python
from lifelines import CoxPHFitter

# 생존 시간 계산
df['duration'] = (df['MCT_ME_D'] - df['ARE_D']).dt.days
df['event'] = df['MCT_ME_D'].notna().astype(int)

# Cox 모델
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event')
cph.print_summary()
```

---

## 4. 모델링 전략

### Phase 1: 베이스라인 모델 (앙상블)

#### 4.1 불균형 데이터 처리
```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks

# Over-sampling
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Hybrid
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
```

#### 4.2 모델 학습
```python
# XGBoost
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)

# LightGBM
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(is_unbalance=True)

# CatBoost
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(auto_class_weights='Balanced')
```

#### 4.3 평가 지표
```python
from sklearn.metrics import f1_score, precision_recall_curve, auc

# F1-score, PR-AUC (Recall 중시)
f1 = f1_score(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_proba)
pr_auc = auc(recall, precision)

# Precision@K (상위 K개 위험 가맹점)
top_k = 100
precision_at_k = precision_score(y_true[:top_k], y_pred[:top_k])
```

---

### Phase 2: 시계열 특성 추가

#### 4.4 시계열 Feature Engineering
```python
# 이동평균
df['sales_ma_3m'] = df.groupby('ENCODED_MCT')['sales'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df['sales_ma_6m'] = df.groupby('ENCODED_MCT')['sales'].transform(
    lambda x: x.rolling(6, min_periods=1).mean()
)

# 증감률
df['sales_mom'] = df.groupby('ENCODED_MCT')['sales'].pct_change()
df['sales_yoy'] = df.groupby('ENCODED_MCT')['sales'].pct_change(12)

# 변동성
df['sales_std_3m'] = df.groupby('ENCODED_MCT')['sales'].transform(
    lambda x: x.rolling(3, min_periods=1).std()
)

# Lag features
for lag in [1, 3, 6]:
    df[f'sales_lag_{lag}m'] = df.groupby('ENCODED_MCT')['sales'].shift(lag)
```

#### 4.5 시계열 모델
```python
# LSTM 데이터 준비
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# XGBoost with time features
df['month'] = df['TA_YM'].astype(str).str[-2:].astype(int)
df['year'] = df['TA_YM'].astype(str).str[:4].astype(int)
df['quarter'] = (df['month'] - 1) // 3 + 1
```

---

### Phase 3: 앙상블 & 해석

#### 4.6 스태킹 앙상블
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Level 1 모델들
estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model),
    ('lstm', lstm_wrapper)
]

# Level 2 모델
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=TimeSeriesSplit(n_splits=5)
)
```

#### 4.7 모델 해석
```python
import shap

# SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 시각화
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
shap.dependence_plot('M1_SME_RY_SAA_RAT', shap_values, X_test)

# Feature importance
import pandas as pd
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 5. Feature Engineering

### 5.1 시계열 특성

#### 추세 및 변화율
- **증감률**:
  - 월간(MoM), 연간(YoY) 매출/고객 증감률
  - 구간 변화 (상승/하락/유지)
- **이동평균**:
  - 3개월, 6개월 이동평균
  - 현재값 - 이동평균 (추세 이탈도)
- **변동성**:
  - 매출/고객 수 표준편차 (3/6개월)
  - 변동계수(CV)

#### 연속성 지표
```python
# 연속 하락 개월 수
df['consecutive_decline'] = df.groupby('ENCODED_MCT')['sales_change'].apply(
    lambda x: (x < 0).astype(int).groupby((x >= 0).cumsum()).cumsum()
)

# 최근 N개월 하락 횟수
df['decline_count_3m'] = df.groupby('ENCODED_MCT')['sales_change'].transform(
    lambda x: (x.rolling(3) < 0).sum()
)
```

---

### 5.2 상대적 지표

#### 경쟁 지표
- **순위 변화**:
  - 업종 내 순위 변화량 (delta)
  - 상권 내 순위 변화량
- **격차 지표**:
  - 업종 평균 대비 격차 (100% 기준)
  - 해지율 차이 (업종/상권 평균 - 자사)

```python
# 순위 변화
df['rank_change_industry'] = df.groupby('ENCODED_MCT')['M12_SME_RY_SAA_PCE_RT'].diff()
df['rank_change_district'] = df.groupby('ENCODED_MCT')['M12_SME_BZN_SAA_PCE_RT'].diff()

# 격차 지표
df['gap_from_industry_avg'] = df['M1_SME_RY_SAA_RAT'] - 100
```

---

### 5.3 고객 행동 지표

#### 충성도 지표
- **재방문율 추세**:
  - 3/6개월 재방문율 변화
  - 재방문율 급락 여부 (임계값 설정)
- **신규 고객 유입**:
  - 신규 고객 비율 변화
  - 신규 유입 감소 지속 개월 수

#### 고객층 변화
```python
# 고객 다양성 (Shannon Entropy)
age_gender_cols = ['M12_MAL_1020_RAT', 'M12_MAL_30_RAT', ...]
df['customer_diversity'] = df[age_gender_cols].apply(
    lambda row: entropy(row / row.sum()), axis=1
)

# 주요 고객층 비중
df['main_segment_ratio'] = df[age_gender_cols].max(axis=1)
```

---

### 5.4 복합 지표

#### 매출-고객 괴리도
```python
# 객단가 변화율 vs 고객 수 변화율
df['price_customer_gap'] = (
    df['price_change_pct'] - df['customer_change_pct']
)

# 매출 하락 시 고객 수 유지 (위험 신호)
df['sales_down_customer_stable'] = (
    (df['sales_change'] < -0.1) & (abs(df['customer_change']) < 0.05)
).astype(int)
```

#### 구간 하락 속도
```python
# 구간 인코딩
interval_mapping = {
    '1_10%이하': 1, '2_10-25%': 2, '3_25-50%': 3,
    '4_50-75%': 4, '5_75-90%': 5, '6_90%초과': 6
}

df['sales_interval_encoded'] = df['RC_M1_SAA'].map(interval_mapping)
df['interval_decline_speed'] = df.groupby('ENCODED_MCT')['sales_interval_encoded'].diff()
```

#### 배달 의존도 변화
```python
# 배달 비율 변화
df['delivery_ratio_change'] = df.groupby('ENCODED_MCT')['DLV_SAA_RAT'].diff()

# 배달 의존도 증가 + 총 매출 감소 (위험 신호)
df['delivery_dependency_risk'] = (
    (df['delivery_ratio_change'] > 10) & (df['sales_change'] < 0)
).astype(int)
```

---

### 5.5 도메인 특화 지표

#### 업종별 위험 지표
```python
# 업종별 평균 폐업률
industry_closure_rate = df.groupby('HPSN_MCT_ZCD_NM')['is_closed'].mean()
df['industry_risk'] = df['HPSN_MCT_ZCD_NM'].map(industry_closure_rate)

# 상권별 경쟁 강도
district_competition = df.groupby(['HPSN_MCT_BZN_CD_NM', 'TA_YM']).size()
df['district_competition'] = df.set_index(['HPSN_MCT_BZN_CD_NM', 'TA_YM']).index.map(district_competition)
```

#### 운영 기간 지표
```python
# 개설 후 경과 개월 수
df['months_since_opening'] = (
    pd.to_datetime(df['TA_YM'], format='%Y%m') -
    pd.to_datetime(df['ARE_D'], format='%Y%m%d')
).dt.days / 30

# 초기 위험 기간 (개설 후 6개월)
df['early_stage_risk'] = (df['months_since_opening'] <= 6).astype(int)
```

---

## 6. 평가 전략

### 6.1 시간 기반 검증

```python
# Train-Valid-Test Split (시계열 순서 유지)
train_end = '202406'
valid_end = '202409'

train_data = df[df['TA_YM'] <= train_end]
valid_data = df[(df['TA_YM'] > train_end) & (df['TA_YM'] <= valid_end)]
test_data = df[df['TA_YM'] > valid_end]

# 시계열 교차 검증
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=3)  # 3개월씩 검증
```

**중요**: 미래 정보 누출(data leakage) 방지
- 특성 생성 시 미래 데이터 사용 금지
- Lag features만 사용
- 시간 순서 엄격히 준수

---

### 6.2 조기 경보 성능 평가

#### 폐업 1~3개월 전 예측 정확도
```python
# 폐업 1개월 전 예측
closure_1m_before = df[
    (df['MCT_ME_D'] - pd.to_datetime(df['TA_YM'], format='%Y%m')).dt.days.between(0, 30)
]

# 성공률 계산
early_warning_rate = (
    (y_pred[closure_1m_before.index] == 1).sum() / len(closure_1m_before)
)
```

#### False Positive 최소화
```python
# Precision-Recall 트레이드오프
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

# 높은 Precision 유지 (불필요한 경고 방지)
target_precision = 0.7
threshold_idx = np.where(precisions >= target_precision)[0][-1]
optimal_threshold = thresholds[threshold_idx]
```

---

### 6.3 비즈니스 지표

#### Top K% 위험군 적중률
```python
# 상위 10% 예측 확률 가맹점
top_k_pct = 0.1
n_top = int(len(y_proba) * top_k_pct)
top_k_indices = np.argsort(y_proba)[-n_top:]

# 실제 폐업 비율
hit_rate = y_true[top_k_indices].mean()
print(f"Top {top_k_pct*100}% 위험군 실제 폐업률: {hit_rate:.2%}")
```

#### 실제 개입 가능 시점
```python
# Lead time 분석 (경고 ~ 폐업 시간 간격)
df['warning_date'] = df['TA_YM']
df['lead_time_days'] = (df['MCT_ME_D'] - pd.to_datetime(df['warning_date'], format='%Y%m')).dt.days

# 평균 리드 타임
avg_lead_time = df[df['prediction'] == 1]['lead_time_days'].mean()
print(f"평균 조기 경보 리드 타임: {avg_lead_time:.0f}일")
```

---

### 6.4 평가 지표 종합

| 지표 | 목적 | 목표값 |
|------|------|--------|
| **PR-AUC** | 전체 성능 | > 0.5 |
| **Recall@k** | 위험 가맹점 탐지율 | > 0.7 |
| **Precision@k** | 경고 정확도 | > 0.6 |
| **F2-score** | Recall 중시 | > 0.6 |
| **Lead Time** | 조기 경보 시점 | > 60일 |

```python
from sklearn.metrics import fbeta_score

# F2-score (Recall을 Precision보다 2배 중요하게)
f2 = fbeta_score(y_true, y_pred, beta=2)

# Custom metric
def early_warning_score(y_true, y_pred, lead_times):
    recall = recall_score(y_true, y_pred)
    avg_lead = lead_times[y_pred == 1].mean()
    return recall * (avg_lead / 90)  # 3개월 기준 정규화
```

---

## 7. 최종 추천

### 7.1 Primary 접근법

**XGBoost/LightGBM + 시계열 Feature Engineering**

**이유**:
1. **해석 가능성 우수** (평가 기준의 인사이트/실효성)
2. **불균형 데이터 강건**
3. **빠른 실험 iteration**
4. **Feature importance로 위기 신호 식별 용이**

**구현 로드맵**:
```
1주차: 데이터 전처리 + 기본 Feature
2주차: 시계열 Feature Engineering
3주차: XGBoost/LightGBM 튜닝
4주차: SHAP 분석 + 인사이트 도출
5주차: 비즈니스 제안 연계
```

---

### 7.2 Secondary 접근법

**LSTM (시퀀스 패턴)**

**이유**:
1. 월별 추세 변화 학습
2. 앙상블 다양성 확보
3. 비선형 패턴 포착

**활용**:
- XGBoost 예측 확률을 보조 특성으로 사용
- Stacking ensemble의 base learner

---

### 7.3 Insight 도출 전략

**SHAP + 도메인 분석**

#### 위기 신호 시각화
```python
import matplotlib.pyplot as plt

# SHAP waterfall plot (개별 가맹점)
shap.plots.waterfall(shap_values[idx])

# SHAP dependence plot (변수 간 상호작용)
shap.dependence_plot(
    'M1_SME_RY_SAA_RAT',  # 메인 변수
    shap_values,
    X_test,
    interaction_index='MCT_UE_CLN_REU_RAT'  # 상호작용 변수
)
```

#### 금융상품 제안 근거
```python
# 위험 유형별 분류
def classify_risk_type(row):
    if row['sales_decline_3m'] > 0.3:
        return '매출 급락형'
    elif row['customer_decline_rate'] > 0.2:
        return '고객 이탈형'
    elif row['delivery_dependency'] > 0.8:
        return '배달 의존형'
    else:
        return '종합 위기형'

df['risk_type'] = df.apply(classify_risk_type, axis=1)

# 유형별 맞춤 상품 매칭
risk_product_map = {
    '매출 급락형': '마케팅 지원 대출',
    '고객 이탈형': '고객 리텐션 컨설팅',
    '배달 의존형': '오프라인 강화 프로그램',
    '종합 위기형': '경영 안정화 패키지'
}
```

---

### 7.4 제출 결과물 구성

#### 1. 핵심 인사이트
- 주요 위기 신호 Top 10
- 업종별/상권별 위험 패턴
- 조기 경보 성공 사례

#### 2. 예측 모델
- 최종 앙상블 모델 (XGB+LGBM+LSTM)
- 성능 지표 (PR-AUC, Recall@K)
- 재현 가능한 코드

#### 3. 비즈니스 제안
- 위험 유형별 금융상품 매칭
- 실행 가능한 개입 전략
- ROI 예측 (경고 정확도 × 구제 가능 매출)

---

## 부록: 참고 자료

### 관련 논문
- Imbalanced Classification: "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
- Time Series: "LSTM for Time Series Prediction" (Hochreiter & Schmidhuber, 1997)
- Interpretability: "A Unified Approach to Interpreting Model Predictions" (SHAP, Lundberg 2017)

### 유용한 라이브러리
```python
# 모델링
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tensorflow import keras

# 불균형 처리
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# 해석
import shap
from lime import lime_tabular

# 시계열
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# 평가
from sklearn.metrics import classification_report, confusion_matrix
```

### 평가 체크리스트
- [ ] 모델링의 적합성/완성도 (25점)
- [ ] 데이터 활용/분석력 (25점)
- [ ] 인사이트/실효성 (20점)
- [ ] 맞춤형 금융상품/서비스 제안 (20점)
- [ ] 완성도 (10점)

---

**작성일**: 2025-10-05
**버전**: 1.0
