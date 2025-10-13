# Interval Pattern Feature Engineering 전략

## 문서 정보
- **작성일**: 2025-10-13
- **버전**: 1.0
- **목적**: 구간 변화 패턴 기반 Feature Engineering 전략 및 구현 가이드

---

## 목차
1. [배경 및 동기](#1-배경-및-동기)
2. [데이터 구조 이해](#2-데이터-구조-이해)
3. [Feature Engineering 전략](#3-feature-engineering-전략)
4. [구현 세부사항](#4-구현-세부사항)
5. [예상 효과](#5-예상-효과)
6. [사용 방법](#6-사용-방법)

---

## 1. 배경 및 동기

### 1.1 기존 Feature Engineering의 한계

기존 접근법에서는 다음과 같은 feature를 생성했습니다:
- Lag features (1, 3, 6, 12개월)
- Moving averages (3, 6, 12개월)
- Change rates (MoM, QoQ, YoY)
- Trend indicators (선형 회귀 기울기)
- Volatility indicators (표준편차, 변동계수)

**문제점**:
1. **구간 정보의 미활용**: 데이터셋의 핵심 정보인 "구간(interval)" 변화 패턴을 충분히 활용하지 못함
2. **절대값 중심**: 구간 자체는 이미 상대적 순위를 나타내므로 절대값보다 **변화 패턴**이 중요
3. **연속성 간과**: 일시적 변동과 구조적 하락을 구분하지 못함

### 1.2 Interval Pattern의 중요성

**핵심 인사이트**:
> 폐업 위험은 절대적 매출액보다 **상대적 순위의 지속적 하락**과 강하게 연관됨

예시:
```
가맹점 A: 매출 구간 2 → 2 → 3 → 3 → 4 (점진적 하락, 위험 ↑)
가맹점 B: 매출 구간 3 → 2 → 4 → 3 → 3 (변동성 높지만 회복, 위험 ↓)
```

→ **연속 하락 패턴**이 폐업 위험의 더 강력한 신호

---

## 2. 데이터 구조 이해

### 2.1 Interval 변수

Dataset 2에는 6개 구간으로 범주화된 변수들이 포함되어 있습니다:

| 구간 코드 | 의미 | 해석 |
|----------|------|------|
| 1 | 상위 10% 이하 | 최상위 성과 |
| 2 | 10-25% | 우수 |
| 3 | 25-50% | 중상위 |
| 4 | 50-75% | 중하위 |
| 5 | 75-90% | 하위 |
| 6 | 90% 초과 | 최하위 성과 |

### 2.2 주요 Interval 변수

```python
interval_columns = [
    'RC_M1_SAA',         # 월 매출액 구간
    'RC_M1_TO_UE_CT',    # 총 이용 건수 구간
    'RC_M1_UE_CUS_CN',   # 이용 고객 수 구간
    'RC_M1_AV_NP_AT'     # 평균 결제 금액 구간
]
```

### 2.3 Interval 변화의 의미

```python
# Interval 증가 = 성과 하락 (1 → 2 → 3 ...)
interval_change > 0  # 하락 (위험 ↑)
interval_change < 0  # 개선 (위험 ↓)
interval_change = 0  # 유지
```

---

## 3. Feature Engineering 전략

### 3.1 Feature 카테고리

#### A. 구간 하락 패턴 (Decline Patterns)

**목적**: 성과 하락의 빈도, 속도, 지속성 포착

**Features**:

1. **Month-over-month interval change**
   ```python
   RC_M1_SAA_interval_change = current_interval - previous_interval
   # positive = decline, negative = improvement
   ```

2. **Decline flag**
   ```python
   RC_M1_SAA_is_declining = 1 if interval_change > 0 else 0
   ```

3. **Consecutive decline count**
   ```python
   RC_M1_SAA_consecutive_declines
   # 연속 하락 개월 수 (예: 3개월 연속 하락 → 3)
   ```

4. **Decline count in window**
   ```python
   RC_M1_SAA_decline_count_3m   # 최근 3개월 중 하락한 개월 수
   RC_M1_SAA_decline_count_6m   # 최근 6개월 중 하락한 개월 수
   RC_M1_SAA_decline_count_12m  # 최근 12개월 중 하락한 개월 수
   ```

5. **Total interval decline**
   ```python
   RC_M1_SAA_total_decline_3m  # 3개월 전 대비 총 구간 하락
   RC_M1_SAA_total_decline_6m  # 6개월 전 대비 총 구간 하락
   RC_M1_SAA_total_decline_12m # 12개월 전 대비 총 구간 하락
   ```

6. **Decline speed**
   ```python
   RC_M1_SAA_decline_speed_3m = total_decline_3m / 3
   # 월 평균 구간 하락 속도
   ```

**예상 효과**:
- 연속 하락은 일시적 변동보다 구조적 문제를 시사 → **폐업 위험의 강력한 신호**
- 하락 속도는 위기의 급박성을 나타냄

---

#### B. 역사적 최악 지표 (Historical Worst Indicators)

**목적**: 과거 대비 현재 성과의 상대적 위치 파악

**Features**:

1. **Worst interval ever**
   ```python
   RC_M1_SAA_worst_ever = cummax(RC_M1_SAA)
   # 역대 최저 구간 (숫자가 클수록 나쁨)
   ```

2. **Best interval ever**
   ```python
   RC_M1_SAA_best_ever = cummin(RC_M1_SAA)
   # 역대 최고 구간
   ```

3. **Is at worst now**
   ```python
   RC_M1_SAA_at_worst_now = 1 if current == worst_ever else 0
   # 현재 역대 최저 구간에 도달했는지 여부
   ```

4. **Distance from best**
   ```python
   RC_M1_SAA_distance_from_best = current - best_ever
   # 역대 최고 대비 하락 폭 (0 = 역대 최고 유지)
   ```

5. **Months since best**
   ```python
   RC_M1_SAA_months_since_best
   # 역대 최고 구간 이후 경과 개월 수
   ```

**예상 효과**:
- 역대 최저 도달은 심리적/실질적 위기 신호
- 최고점 이후 장기간 하락은 회복 불가능 위험 시사

---

#### C. 회복 지표 (Recovery Indicators)

**목적**: 하락 후 회복 능력 및 변동성 평가

**Features**:

1. **Recovery flag**
   ```python
   RC_M1_SAA_is_recovering = 1 if interval_change < 0 else 0
   # 구간 개선 여부
   ```

2. **Consecutive recovery count**
   ```python
   RC_M1_SAA_consecutive_recovery
   # 연속 회복 개월 수
   ```

3. **Recovery after decline**
   ```python
   RC_M1_SAA_recovery_after_decline
   # 하락 직후 회복 여부 (resilience 지표)
   ```

4. **Interval volatility**
   ```python
   RC_M1_SAA_interval_volatility_3m  # 3개월 구간 변화의 표준편차
   RC_M1_SAA_interval_volatility_6m  # 6개월 구간 변화의 표준편차
   ```

5. **Direction changes**
   ```python
   RC_M1_SAA_direction_changes_3m
   # 방향 전환 횟수 (상승↔하락 oscillation)
   ```

**예상 효과**:
- 회복 패턴은 가맹점의 resilience 반영
- 높은 변동성은 불안정성을 나타냄 → 위험 요소

---

#### D. 교차 지표 (Cross-Metric Indicators)

**목적**: 서로 다른 지표 간 divergence로 위기 원인 파악

**Features**:

1. **Divergence: Sales declining but customers stable**
   ```python
   divergence_RC_M1_SAA_vs_RC_M1_UE_CUS_CN
   # 매출은 하락하지만 고객 수는 유지 → 가격/품질 문제 시사
   ```

2. **Aligned decline: Both declining**
   ```python
   aligned_decline_RC_M1_SAA_RC_M1_UE_CUS_CN
   # 매출+고객 수 동시 하락 → 치명적 위험 신호
   ```

3. **Divergence magnitude**
   ```python
   divergence_magnitude_RC_M1_SAA_RC_M1_UE_CUS_CN
   # 매출 변화 - 고객 수 변화 (괴리 정도)
   ```

**예상 효과**:
- Divergence 패턴으로 위기의 **근본 원인** 진단 가능
- 맞춤형 개입 전략 수립에 활용

**패턴 예시**:

| 매출 구간 | 고객 구간 | 해석 | 개입 전략 |
|----------|----------|------|----------|
| 하락 ↓ | 유지 → | 객단가 하락, 가격 경쟁력 문제 | 프리미엄 메뉴 개발, 가격 전략 재검토 |
| 유지 → | 하락 ↓ | 소수 고객 의존도 증가 위험 | 고객 다변화, 마케팅 강화 |
| 하락 ↓ | 하락 ↓ | 종합적 위기 (치명적) | 긴급 경영 개입, 구조조정 |
| 개선 ↑ | 하락 ↓ | 객단가 상승하지만 고객 이탈 | 가격 정책 재조정, 고객 유지 프로그램 |

---

## 4. 구현 세부사항

### 4.1 구조

```
pipeline/features/interval_patterns.py
└── IntervalPatternFeatureEngine
    ├── create_interval_decline_features()       # 하락 패턴
    ├── create_historical_worst_features()       # 역사적 최악
    ├── create_recovery_indicators()             # 회복 지표
    ├── create_cross_metric_interval_features()  # 교차 지표
    └── create_all_interval_features()           # 통합 실행
```

### 4.2 사용 예시

```python
from pipeline.features import IntervalPatternFeatureEngine

# Engine 초기화
engine = IntervalPatternFeatureEngine(
    merchant_col='ENCODED_MCT',
    date_col='TA_YM'
)

# Interval 변수 지정
interval_columns = [
    'RC_M1_SAA',         # 매출액 구간
    'RC_M1_TO_UE_CT',    # 이용 건수 구간
    'RC_M1_UE_CUS_CN',   # 고객 수 구간
    'RC_M1_AV_NP_AT'     # 객단가 구간
]

# 모든 interval pattern features 생성
df_with_intervals = engine.create_all_interval_features(
    df,
    interval_columns=interval_columns
)
```

### 4.3 개별 Feature 생성

```python
# 1. 하락 패턴만 생성
df = engine.create_interval_decline_features(
    df,
    interval_columns=interval_columns,
    windows=[3, 6, 12]
)

# 2. 역사적 최악 지표만 생성
df = engine.create_historical_worst_features(
    df,
    interval_columns=interval_columns
)

# 3. 회복 지표만 생성
df = engine.create_recovery_indicators(
    df,
    interval_columns=interval_columns
)

# 4. 교차 지표만 생성
df = engine.create_cross_metric_interval_features(
    df,
    primary_col='RC_M1_SAA',
    secondary_cols=['RC_M1_UE_CUS_CN', 'RC_M1_TO_UE_CT']
)
```

### 4.4 주의사항

#### Data Leakage 방지
- Interval 변화는 **과거 데이터만** 사용 (`shift()`, `diff()` 활용)
- Cumulative features (worst_ever, best_ever)는 **현재 시점까지만** 계산

```python
# ✅ 올바른 예: 이전 달 대비 변화
interval_change = df.groupby('ENCODED_MCT')['RC_M1_SAA'].diff()

# ❌ 잘못된 예: 미래 데이터 포함
worst_ever = df.groupby('ENCODED_MCT')['RC_M1_SAA'].transform('max')  # 전체 기간의 max
```

#### 결측값 처리
- 첫 달은 diff() 결과가 NaN → 자연스러운 현상
- Consecutive count는 초기값 0으로 처리

---

## 5. 예상 효과

### 5.1 성능 개선 예상

| 지표 | 기존 모델 (예상) | Interval Pattern 추가 후 (목표) |
|------|-----------------|-------------------------------|
| **PR-AUC** | 0.45 | **> 0.55** (+0.10) |
| **Recall@10%** | 0.60 | **> 0.70** (+0.10) |
| **Precision@10%** | 0.50 | **> 0.60** (+0.10) |

### 5.2 Feature Importance 예상 순위

**Top 5 예상 Features**:
1. `RC_M1_SAA_consecutive_declines` - 매출 구간 연속 하락 개월 수
2. `aligned_decline_RC_M1_SAA_RC_M1_UE_CUS_CN` - 매출+고객 동시 하락
3. `RC_M1_SAA_at_worst_now` - 역대 최저 구간 도달
4. `RC_M1_SAA_decline_count_3m` - 최근 3개월 하락 횟수
5. `RC_M1_SAA_distance_from_best` - 최고 구간 대비 하락 폭

### 5.3 비즈니스 가치

#### 조기 경보 정확도 향상
- **Lead time 증가**: 평균 60일 → **90일** 목표
- **False positive 감소**: 연속성 지표로 일시적 변동 필터링

#### 위기 유형 분류 가능
```python
if consecutive_declines >= 3 and at_worst_now:
    risk_type = "급격한 구조적 하락 - 긴급 개입 필요"

elif divergence_sales_vs_customers:
    risk_type = "가격/품질 문제 - 상품 전략 재검토"

elif aligned_decline:
    risk_type = "종합 위기 - 경영 전반 점검"
```

#### 맞춤형 금융상품 제안
- **연속 하락형** → 단기 운영자금 대출 + 경영 컨설팅
- **Divergence형** → 마케팅 지원 대출 + 고객 분석
- **최저 구간 도달형** → 긴급 안정화 패키지

---

## 6. 사용 방법

### 6.1 Notebook 실행

```bash
cd notebooks
jupyter notebook 03-1_interval_pattern_features.ipynb
```

### 6.2 Pipeline 통합

기존 feature engineering pipeline에 추가:

```python
# 03_feature_engineering.ipynb 또는 pipeline 스크립트

# 기존 features
df = ts_engine.create_lag_features(df, columns=lag_cols)
df = ts_engine.create_moving_averages(df, columns=ma_cols)
# ...

# ✨ Interval pattern features 추가
from pipeline.features import IntervalPatternFeatureEngine

interval_engine = IntervalPatternFeatureEngine()
df = interval_engine.create_all_interval_features(
    df,
    interval_columns=['RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT']
)

# 저장
df.to_csv('data/processed/featured_data_final.csv', index=False)
```

### 6.3 모델 학습 시 활용

```python
# Feature selection
interval_features = [col for col in df.columns if any(
    kw in col for kw in ['interval_change', 'decline', 'worst', 'recovery', 'divergence']
)]

# XGBoost 학습
X_train = df[df['is_valid_for_training'] == 1][feature_cols + interval_features]
y_train = df[df['is_valid_for_training'] == 1]['will_close_3m']

model = xgb.XGBClassifier(scale_pos_weight=32)
model.fit(X_train, y_train)

# Feature importance 확인
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(importance_df.head(20))
```

### 6.4 SHAP 분석

```python
import shap

# SHAP values 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Interval pattern features만 필터링
interval_feature_indices = [i for i, col in enumerate(X_test.columns)
                            if any(kw in col for kw in ['decline', 'worst', 'divergence'])]

# Summary plot
shap.summary_plot(
    shap_values[:, interval_feature_indices],
    X_test.iloc[:, interval_feature_indices],
    plot_type='bar'
)
```

---

## 7. 검증 및 모니터링

### 7.1 Feature 품질 검증

```python
# 1. 결측값 확인
missing_ratio = df[interval_features].isnull().sum() / len(df)
print(f"Missing ratio: {missing_ratio[missing_ratio > 0.1]}")  # 10% 이상만 표시

# 2. 분포 확인
for feature in ['RC_M1_SAA_consecutive_declines', 'RC_M1_SAA_decline_count_3m']:
    print(f"\n{feature} distribution:")
    print(df[feature].value_counts().sort_index())

# 3. 타겟과 상관관계
correlations = df[interval_features].corrwith(df['will_close_3m']).abs()
print(f"\nTop 10 correlations with target:")
print(correlations.sort_values(ascending=False).head(10))
```

### 7.2 폐업 예정 vs 정상 비교

```python
comparison = df.groupby('will_close_3m')[interval_features].mean().T
comparison.columns = ['Normal', 'Will Close']
comparison['Difference'] = comparison['Will Close'] - comparison['Normal']
comparison = comparison.sort_values('Difference', ascending=False)

print("\nFeature differences (Will Close vs Normal):")
print(comparison.head(10))
```

### 7.3 시계열 검증

```python
# Train/Valid/Test split으로 시간 기반 검증
train_dates = df['TA_YM'] <= 202406
valid_dates = (df['TA_YM'] > 202406) & (df['TA_YM'] <= 202409)
test_dates = df['TA_YM'] > 202409

# 각 기간별 feature 분포 확인
for period, mask in [('Train', train_dates), ('Valid', valid_dates), ('Test', test_dates)]:
    print(f"\n{period} period:")
    print(df[mask]['RC_M1_SAA_consecutive_declines'].describe())
```

---

## 8. 향후 확장 가능성

### 8.1 추가 Interval Features (Phase 2)

1. **계절성 조정 구간 변화**
   ```python
   # 동일 분기 전년 대비 구간 변화
   interval_change_yoy = current_interval - interval_12m_ago
   ```

2. **업종/상권 내 상대 구간**
   ```python
   # 동일 업종 내 구간 순위
   industry_interval_rank = rank(interval) within industry
   ```

3. **구간 변화 가속도**
   ```python
   # 2차 미분 (변화율의 변화)
   interval_acceleration = (change_1m - change_2m)
   ```

### 8.2 고급 패턴 인식

1. **Regime change detection**
   - Hidden Markov Model로 "안정 → 하락" 전환점 탐지

2. **Pattern matching**
   - Dynamic Time Warping (DTW)로 유사 하락 패턴 가맹점 군집화

3. **Survival analysis integration**
   - Interval 변화를 Cox model의 time-varying covariate로 활용

---

## 9. 참고 자료

### 9.1 관련 코드
- `pipeline/features/interval_patterns.py` - Feature engine 구현
- `notebooks/03-1_interval_pattern_features.ipynb` - Feature 생성 및 분석
- `docs/11_data_leakage_prevention.md` - Data leakage 방지 가이드

### 9.2 관련 논문
- **Sequence Pattern Mining**: "Sequential Pattern Mining: A Survey" (Han et al., 2007)
- **Time Series Patterns**: "Mining Sequential Patterns" (Agrawal & Srikant, 1995)
- **Business Failure Prediction**: "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy" (Altman, 1968)

### 9.3 유사 사례
- **신용 위험 평가**: 신용등급 하락 패턴 분석
- **고객 이탈 예측**: 사용량 구간 변화 추적
- **재무 건전성 평가**: 재무비율 구간 추세 분석

---

## 부록: Feature 전체 목록

### A. Decline Pattern Features (각 interval column당 생성)

```
{column}_interval_change              # MoM 구간 변화
{column}_is_declining                 # 하락 flag
{column}_consecutive_declines         # 연속 하락 개월 수
{column}_decline_count_3m            # 3개월 하락 횟수
{column}_decline_count_6m            # 6개월 하락 횟수
{column}_decline_count_12m           # 12개월 하락 횟수
{column}_total_decline_3m            # 3개월 전 대비 총 하락
{column}_total_decline_6m            # 6개월 전 대비 총 하락
{column}_total_decline_12m           # 12개월 전 대비 총 하락
{column}_decline_speed_3m            # 3개월 평균 하락 속도
{column}_decline_speed_6m            # 6개월 평균 하락 속도
```

### B. Historical Worst Features

```
{column}_worst_ever                   # 역대 최저 구간
{column}_best_ever                    # 역대 최고 구간
{column}_at_worst_now                 # 현재 최저 여부
{column}_distance_from_best           # 최고 대비 하락 폭
{column}_months_since_best            # 최고 이후 경과 개월
```

### C. Recovery Features

```
{column}_is_recovering                # 회복 flag
{column}_consecutive_recovery         # 연속 회복 개월 수
{column}_recovery_after_decline       # 하락 후 회복 여부
{column}_interval_volatility_3m       # 3개월 변동성
{column}_interval_volatility_6m       # 6개월 변동성
{column}_direction_changes_3m         # 3개월 방향 전환 횟수
{column}_direction_changes_6m         # 6개월 방향 전환 횟수
```

### D. Cross-Metric Features

```
divergence_{primary}_{secondary}                    # Divergence flag
aligned_decline_{primary}_{secondary}               # 동시 하락 flag
divergence_magnitude_{primary}_{secondary}          # Divergence 크기
```

**Total**: 약 **100+ features** (4개 interval column 기준)

---

**작성자**: Claude Code
**최종 수정**: 2025-10-13
**버전**: 1.0
