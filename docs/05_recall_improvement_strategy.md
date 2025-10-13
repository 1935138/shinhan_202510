# Recall 개선 전략 (Recall Improvement Strategy)

## 문제 정의

### 초기 성능 문제
- **Test Recall: 4.5%** (22개 폐업 예정 가맹점 중 1개만 탐지)
- **Test Precision: 9.1%**
- **Test F1-Score: 6.1%**

조기 경보 시스템에서 Recall 4.5%는 치명적인 문제:
- 폐업 예정 가맹점 95%를 놓침
- 실제 위기 상황의 대부분을 탐지하지 못함
- 비즈니스 가치가 없는 수준

### 목표
- **Target Recall: 60%+** (최소 10개 이상/22개 탐지)
- **Acceptable Precision: 30%+** (False Positive를 어느 정도 감수)
- **F2-Score 최대화** (Recall을 Precision보다 2배 중시)

---

## 근본 원인 분석

### 1. 극심한 클래스 불균형 (Extreme Class Imbalance)

**데이터 분포**:
```
Train:      45 positive / 39,778 total (0.11%)
Validation: 29 positive / 22,479 total (0.13%)
Test:       22 positive / 24,303 total (0.09%)
```

**문제점**:
- 양성 클래스 비율이 0.1%로 극도로 낮음
- 모델이 "항상 음성으로 예측"하는 전략을 학습
- 99.9% 정확도를 달성하면서도 양성 샘플을 하나도 맞추지 못함

### 2. 낮은 예측 확률

**Baseline 모델의 예측 확률 분포**:
```
Validation - Mean: 0.0008 (0.08%)
Test       - Mean: 0.0008 (0.08%)
Max probability: 0.59 (Val), 0.44 (Test)
```

**문제점**:
- 평균 예측 확률이 0.08%로 극도로 낮음
- 모델이 거의 모든 샘플을 음성으로 확신
- Threshold를 아무리 낮춰도 양성으로 분류되는 샘플이 거의 없음

### 3. Data Leakage 수정의 부작용

이전에는 `is_closed` 변수를 사용하여 100% data leakage가 있었고, 이로 인해 높은 성능을 보였음:
- Old (leakage 있음): ROC-AUC 92.6%, Recall ~80%+
- New (leakage 수정): ROC-AUC 75.2%, Recall 4.5%

Data leakage를 제거하고 `will_close_3m`을 사용하면서 **현실적이지만 매우 낮은 성능**이 드러남.

---

## 시도한 개선 전략

### 전략 1: Aggressive Scale Positive Weight

**방법**:
```python
# Original scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# = 39,733 / 45 = 882.96

# Aggressive: 2배 증가
aggressive_scale = scale_pos_weight * 2.0
# = 1,765.91

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=aggressive_scale,
    # ... other params
)
```

**목적**:
- 양성 샘플의 loss weight를 2배로 증가
- 모델이 양성 클래스를 더 중요하게 학습하도록 유도

**결과**:
- Validation Recall: 여전히 매우 낮음
- 예측 확률 분포: 거의 변화 없음 (Mean 0.0008)
- **실패**: scale_pos_weight만으로는 극심한 불균형 해결 불가

### 전략 2: Classification Threshold 최적화

**방법**:
```python
# F2-score를 최대화하는 threshold 탐색
thresholds = np.linspace(0.01, 0.99, 99)
best_threshold = 0.5
best_f2 = 0

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f2 = fbeta_score(y_val, y_pred, beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = threshold
```

**목적**:
- 기본 threshold 0.5를 낮춰서 더 많은 양성 예측 생성
- F2-score 최대화 (Recall에 가중치)

**결과**:
- **Best threshold: 0.07** (0.5 → 0.07로 대폭 하락)
- Validation:
  - Recall: 6.9% (2/29 탐지)
  - Precision: 7.4%
  - F2-Score: 0.0699
- Test:
  - Recall: **0%** (0/22 탐지)
  - Precision: 0%
  - F2-Score: 0

**문제점**:
- Threshold를 0.07까지 낮춰도 거의 효과 없음
- 근본 원인: 예측 확률 자체가 너무 낮음 (대부분 0.01 이하)
- **부분 성공**: Validation에서 2개 탐지했으나 Test에서는 완전 실패

### 전략 3: SMOTE Oversampling ✅

**방법**:
```python
from imblearn.over_sampling import SMOTE

# SMOTE with sampling_strategy=0.3
# 양성 클래스를 음성 클래스의 30%로 증가
smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Before: 45 positive / 39,778 total (0.11%)
# After:  11,919 positive / 51,652 total (23.08%)
# Synthetic samples created: 11,874
```

**원리**:
- 기존 양성 샘플의 k-nearest neighbors를 이용해 합성 샘플 생성
- 양성 클래스의 feature space를 확장
- 모델이 양성 패턴을 더 잘 학습할 수 있도록 함

**실제 결과**:

**긍정적 효과**:
- ✅ **예측 확률 대폭 증가**:
  - Validation: 0.0008 → 0.0066 (+729%)
  - Test: 0.0008 → 0.0012 (+53%)
  - Max probability: Val 87.5%, Test 68.5%
- ✅ **Threshold 상승**: 0.07 → 0.26 (예측 확률 증가로 인함)
- ✅ **ROC-AUC 소폭 개선**: Test 0.71 → 0.69 (약간 하락), Val 0.71 → 0.57 (하락)

**부정적 결과**:
- ❌ **Test Recall 여전히 0%**: 22개 중 0개 탐지 (목표 대비 실패)
- ⚠️ **Validation Recall 미미한 개선**: 0% → 3.4% (1/29 탐지)
- ⚠️ **Validation-Test 성능 차이 심각**: Val에서 1개 탐지했지만 Test에서 0개
- ⚠️ **Precision 하락**: Val 7.4% → 1.6%, Test 0% → 0%

**Confusion Matrix**:
```
Validation (29개 양성):
  TP: 1, FN: 28, FP: 63, TN: 22,387
  Detection Rate: 3.4%

Test (22개 양성):
  TP: 0, FN: 22, FP: 20, TN: 24,261
  Detection Rate: 0%
```

**결론**:
- **부분 성공**: 예측 확률 분포 개선에는 성공
- **주요 실패**: Recall 개선에는 실패 (Test 0% 유지)
- **Overfitting 의심**: Validation에서 1개 탐지했으나 Test에서 완전 실패
- **근본 원인**: Feature 자체가 폐업 신호를 충분히 포착하지 못하는 것으로 보임

---

## 전략 비교표

| 전략 | 구현 난이도 | Training 시간 | Val Recall | Test Recall | Val Precision | Test Precision | 비고 |
|------|------------|--------------|------------|-------------|---------------|----------------|------|
| **Baseline (scale_pos_weight)** | 쉬움 | 빠름 | 0% | 0% | 0% | 0% | 기준선 |
| **Aggressive scale_pos_weight** | 쉬움 | 빠름 | 0% | 0% | 0% | 0% | 효과 없음 |
| **Threshold 최적화 (T=0.07)** | 쉬움 | 빠름 | **6.9%** | **0%** | 7.4% | 0% | Val에서만 부분 성공 |
| **SMOTE (sampling=0.3)** | 보통 | 중간 | 3.4% | **0%** | 1.6% | 0% | ❌ 실패 (Test 0%) |
| **Feature Engineering** | 어려움 | 느림 | - | - | - | - | **다음 시도 필요** |
| **Ensemble 모델** | 보통 | 느림 | - | - | - | - | 삭제됨 (효과 없음) |

**핵심 발견**:
- 모든 전략에서 **Test Recall 0%** 유지
- Threshold 최적화가 유일하게 Val에서 6.9% 달성 (2/29 탐지)
- SMOTE는 예측 확률 증가시켰지만 Recall은 오히려 하락 (6.9% → 3.4%)

---

## 다음 단계 (Next Steps)

### 1. SMOTE 결과 분석 (우선순위: 높음)
- [ ] Cell 30-37 실행하여 SMOTE 효과 검증
- [ ] 예측 확률 분포 변화 확인
- [ ] Validation/Test Recall 개선 여부 확인
- [ ] Confusion Matrix 분석

### 2. Hyperparameter Tuning (우선순위: 중간)
**현재 계획 (Todo #4)**:
```python
# Recall-focused hyperparameter tuning
param_grid = {
    'max_depth': [3, 4, 5, 6],           # 얕은 트리로 일반화
    'min_child_weight': [1, 5, 10],      # 양성 샘플 분할 기준 완화
    'learning_rate': [0.01, 0.05, 0.1],  # 낮은 학습률
    'n_estimators': [200, 300, 500],     # 더 많은 트리
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Use PR-AUC or Recall@K as scoring metric
RandomizedSearchCV(
    xgb_model,
    param_grid,
    scoring='average_precision',  # PR-AUC
    cv=TimeSeriesSplit(n_splits=3)
)
```

**목표**:
- SMOTE 모델에 최적화된 하이퍼파라미터 찾기
- PR-AUC 또는 F2-score 최대화

### 3. Feature Engineering (우선순위: 높음)
**현재 문제점**:
- 168개 feature 중 많은 feature가 결측값 포함 (55%+)
- Lag features (12개월)가 많아 초기 데이터 손실
- Time-series features가 폐업 신호를 충분히 포착하지 못함

**개선 방안**:
- [ ] **더 짧은 lag window**: 12개월 → 3-6개월로 단축 (데이터 손실 감소)
- [ ] **Aggregate features**: 매출/고객 관련 feature의 복합 지표 생성
  - 매출-고객 괴리도 (매출 감소 but 고객 유지)
  - 배달 의존도 위험 (배달 매출 > 70% + 총 매출 감소)
- [ ] **Trend features**: 연속 하락 개월 수, 하락 가속도
- [ ] **상권 features**: 같은 상권 내 폐업률, 경쟁 강도
- [ ] **Feature selection**: 중요도 낮은 feature 제거 (168개 → 80-100개)

### 4. Alternative Sampling 기법 (우선순위: 낮음)
- [ ] **ADASYN**: SMOTE의 개선 버전 (경계 영역에 더 많은 샘플 생성)
- [ ] **Borderline-SMOTE**: 분류 경계 근처 샘플만 oversampling
- [ ] **SMOTE + Undersampling**: 양쪽 클래스 조정
- [ ] **Different sampling_strategy**: 0.1, 0.5, 0.7 등 실험

### 5. Alternative Models (우선순위: 낮음)
- [ ] **CatBoost**: 불균형 데이터 처리에 강함
- [ ] **Balanced Random Forest**: 내장 균형 처리
- [ ] **Cost-sensitive learning**: Focal Loss 등

### 6. Evaluation Metric 재검토 (우선순위: 높음)
**현재 평가 지표**:
- Recall, Precision, F1, F2, ROC-AUC, PR-AUC

**비즈니스 관점 지표 추가**:
- [ ] **Recall@K**: 상위 K%에서 몇 개 탐지? (조기 경보 관점)
  - 예: Recall@10% = 상위 10% 위험군에서 실제 폐업의 몇 %를 포함?
- [ ] **Precision@K**: 상위 K% 중 실제 폐업 비율
  - 예: Precision@10% = 상위 10% 중 실제 폐업 비율
- [ ] **Average Lead Time**: 폐업 전 평균 몇 개월 전에 경보?
- [ ] **Cost-based metric**: False Negative의 비용 > False Positive의 비용

---

## 핵심 인사이트

### 1. 단일 기법으로는 해결 불가
- scale_pos_weight, threshold 최적화 각각으로는 불충분
- **복합 전략 필요**: SMOTE + Threshold 최적화 + Feature Engineering

### 2. 예측 확률이 핵심
- Threshold를 아무리 낮춰도 예측 확률 자체가 낮으면 무용
- **SMOTE의 핵심 목표**: 예측 확률 분포를 높이는 것

### 3. Validation-Test 성능 차이
- Validation Recall 6.9% but Test Recall 0%
- **과적합 또는 데이터 분포 차이** 의심
- Train: 2023년, Val: 2024 H1, Test: 2024 H2
- 시간에 따른 폐업 패턴 변화 가능성

### 4. 비즈니스 맥락 중요
- Recall 60%가 달성 가능한 목표인가?
- 실제 폐업 신호가 데이터에 포함되어 있는가?
- **데이터의 근본적 한계**: 22개 Test 샘플로는 안정적 평가 어려움

### 5. 데이터 부족 문제
- Test set에 22개 양성 샘플만 존재
- 1개만 더 맞춰도 Recall이 4.5% → 9.1%로 2배 증가
- **통계적으로 불안정**: 더 많은 데이터 필요

---

## 결론 및 최종 권장사항

### 현재 상황 (2025-10-13 업데이트)

**시도한 모든 전략 실패**:
- ❌ Aggressive scale_pos_weight: Test Recall 0%
- ❌ Threshold 최적화 (T=0.07): Val 6.9%, Test 0%
- ❌ SMOTE (sampling=0.3): Val 3.4%, Test 0%

**핵심 발견**:
1. **Test Set에서 단 하나도 탐지 못함**: 3가지 전략 모두 Test Recall 0%
2. **Validation-Test 성능 차이 극심**: Val에서 미미하게 작동해도 Test에서 완전 실패
3. **예측 확률 개선만으로는 불충분**: SMOTE가 확률을 729% 증가시켰지만 Recall은 오히려 하락

### 근본적인 문제 진단

#### 1. Feature의 예측력 부족 (가장 치명적)
- 현재 154개 features (14개 all-NaN 제거 후)
- 11,874개 합성 샘플(SMOTE)로 학습해도 성능 향상 없음
- **결론**: Feature 자체가 폐업 3개월 전 신호를 포착하지 못함

#### 2. 데이터의 본질적 한계
- Test set 양성: 단 22개 (통계적으로 불안정)
- Train: 2023년, Val: 2024 H1, Test: 2024 H2
- 시간에 따른 폐업 패턴 변화 가능성

#### 3. 극심한 불균형
- Training 양성 비율: 0.11% (45/39,778)
- SMOTE로 23%까지 증가시켰지만 여전히 불충분

### 🚨 긴급 권장사항

**더 이상 Sampling/Threshold 최적화로는 해결 불가능. Feature Engineering이 필수.**

#### 우선순위 1: Feature Engineering (즉시 시작)
```python
# 제거해야 할 features
- lag_12m features (14개 all-NaN)
- 결측률 50% 이상 features

# 추가해야 할 features (폐업 신호 포착)
1. 더 짧은 시간 창 (3개월, 6개월)
   - sales_decline_3m: 3개월 연속 매출 하락 여부
   - customer_decline_6m: 6개월 재방문율 추세

2. 복합 지표
   - 매출-고객 괴리도: (매출 변화율 - 고객 변화율)
   - 배달 의존도 위험: delivery_ratio > 0.7 AND total_sales_down

3. 상대적 위치
   - 업종 내 순위 변화
   - 상권 내 상대 매출

4. 가속도 지표
   - 매출 감소 가속도 (2차 미분)
   - 연속 하락 개월 수
```

#### 우선순위 2: Alternative 접근법
1. **더 강력한 Sampling**
   - sampling_strategy=0.5, 0.7, 1.0 시도
   - ADASYN, BorderlineSMOTE

2. **Class Weight 극대화**
   - scale_pos_weight=5000, 10000 (현재 882의 5-10배)

3. **Anomaly Detection 전환**
   - 폐업 예측을 이상 탐지 문제로 재정의
   - Isolation Forest, One-Class SVM

#### 우선순위 3: 데이터 확보 (장기)
- 2022년 이전 데이터 확보
- 외부 데이터 결합 (경제 지표, 부동산 가격, 인구 통계)
- 다른 지역 데이터 확보

### 현실적 목표 재설정

**기존 목표 (달성 불가능)**:
- ~~단기: Test Recall 30-40%~~
- ~~중기: Test Recall 60%+~~
- ~~장기: PR-AUC 0.55+~~

**수정된 목표 (현실적)**:
- **즉시**: Feature engineering 후 Test Recall 10-20% 달성
- **단기**: 새로운 features로 Test Recall 30%+ 달성
- **중기**: 외부 데이터 결합 후 Test Recall 50%+ 달성
- **장기**: 비즈니스적으로 의미 있는 조기 경보 시스템 구축

### 다음 액션 아이템

1. **Feature engineering 우선** (`notebooks/03_feature_engineering.ipynb` 수정)
   - [ ] lag_12m 제거, lag_3m/lag_6m 추가
   - [ ] 복합 지표 10개 이상 생성
   - [ ] Feature importance 기반 선택 (상위 80개)

2. **모델 재학습** (`notebooks/04_model_training.ipynb`)
   - [ ] 새 features로 baseline 재평가
   - [ ] SMOTE + 새 features 조합 시도

3. **대안 탐색**
   - [ ] Anomaly detection 접근법 시도
   - [ ] Rule-based 모델과 ML 모델 하이브리드

---

**최종 업데이트**: 2025-10-13
**작성자**: Claude Code
**버전**: 2.0 (SMOTE 실험 완료 반영)
**관련 파일**:
- `notebooks/04_model_training.ipynb` (Section 5.3-5.5)
- `notebooks/03_feature_engineering.ipynb` (다음 수정 대상)
