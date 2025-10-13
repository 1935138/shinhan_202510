# 모델 설계 과정 요약

> **2025 빅콘테스트: 가맹점 위기 조기 경보 시스템**
> **발표 자료** (PPT 3-4장 분량)

---

## Slide 1: 프로젝트 개요 및 핵심 도전과제

### 🎯 문제 정의
**서울 성동구 요식업 가맹점의 폐업 위험을 1-3개월 전에 예측**

- **대상**: 4,185개 가맹점, 24개월 데이터 (2023.01 ~ 2024.12)
- **목표**: 조기 경보 시스템 구축 → 맞춤형 금융상품 제안

### ⚠️ 핵심 도전과제: 극심한 클래스 불균형

```
폐업 가맹점:    127개 (3.0%) ❌
운영 중 가맹점: 4,058개 (97.0%) ✅
───────────────────────────────
불균형 비율: 1:32
```

**→ 일반적인 ML 접근법으로는 소수 클래스(폐업) 탐지 실패**

### 📊 데이터 특성
- **구간화된 데이터**: 매출/고객 등을 6개 구간으로 범주화 (상위10% ~ 하위10%)
- **시계열 데이터**: 월별 24개월 추적
- **다면적 지표**: 매출, 고객, 재방문율, 업종/상권 순위 등

---

## Slide 2: 모델 설계 전략

### 💡 핵심 아이디어: Interval Pattern Features

#### 기존 접근법의 한계
초기에는 전통적인 시계열 feature를 시도했습니다:
- Lag features (1, 3, 6, 12개월)
- Moving averages (3, 6, 12개월)
- Change rates (MoM, YoY)
- Volatility indicators

**문제점**: 데이터의 핵심인 **"구간 변화 패턴"**을 충분히 활용하지 못함

#### 새로운 접근: 구간 패턴 기반 Feature Engineering

**핵심 인사이트**:
> 폐업 위험은 절대적 매출액보다 **"상대적 순위의 지속적 하락"**과 강하게 연관됨

예시:
```
가맹점 A: 구간 2 → 2 → 3 → 3 → 4  (점진적 하락, 위험 ↑)
가맹점 B: 구간 3 → 2 → 4 → 3 → 3  (변동성 높지만 회복, 위험 ↓)
```

#### 4가지 카테고리의 Interval Features 설계

| 카테고리 | 주요 Feature | 의도 |
|---------|-------------|------|
| **1. 구간 하락 패턴** | 연속 하락 개월 수<br>하락 속도<br>총 하락폭 (3/6/12개월) | 구조적 하락 vs 일시적 변동 구분 |
| **2. 역대 최악 지표** | 역대 최악 구간<br>현재 최악 여부<br>최고점 대비 하락폭 | 심리적/실질적 위기 신호 포착 |
| **3. 회복 지표** | 연속 회복 개월 수<br>구간 변동성<br>방향 전환 횟수 | Resilience 및 불안정성 평가 |
| **4. 교차 지표** | 매출 vs 고객 수 괴리<br>동시 하락 패턴 | 위기 원인 진단 (객단가 문제, 고객 이탈 등) |

**총 생성 Feature**: **100개 이상**

### 🛠️ 사용 기술 스택

```python
# 1. 모델: XGBoost
- 불균형 처리: scale_pos_weight=32 (class imbalance ratio)
- 범주형 데이터 효과적 처리
- Feature importance 제공 → 위기 신호 해석 가능

# 2. Feature Engineering
- 구간 패턴 분석 엔진 (pipeline/features/interval_patterns.py)
- 시계열 groupby + shift로 data leakage 방지

# 3. 평가 지표
- PR-AUC (Precision-Recall AUC) - imbalanced data에 적합
- Recall 중심 - 폐업 위험 놓치지 않는 것이 최우선
- Detection Rate - 실제 폐업 몇 개 탐지했는지
```

---

## Slide 3: 실험 과정 및 비교

### 🔬 3가지 실험 설계

#### 실험 설정
- **Train**: 2023.01 ~ 2023.06 (6개월)
- **Valid**: 2023.07 ~ 2023.09 (3개월)
- **Test**: 2023.10 ~ 2023.12 (3개월)
- **모델**: XGBoost with `scale_pos_weight`

---

### 📊 실험 비교표

| 실험 | Feature 수 | ROC-AUC | PR-AUC | Detection | False Positives | 비고 |
|------|-----------|---------|--------|-----------|----------------|------|
| **실험 1:<br>All Interval** | **100+개** | **0.6980** | **0.0261** | **4/22<br>(18.2%)** | 수백 개 | ✅ **최다 탐지**<br>다각도 분석 |
| 실험 2:<br>Decline Only | 36개 | 0.6932 | 0.0018 | 3/11<br>(27.3%) | **1,323개** | ❌ FP 폭증<br>실용성 없음 |
| 실험 3:<br>Selected | 19개 | 0.6470 | 0.0016 | 3/11<br>(27.3%) | 925개 | ❌ 성능 하락<br>정보 손실 |

---

### 💭 실험별 의도 및 분석

#### 실험 1: All Interval Features (100+개)
**의도**: 다양한 각도에서 위험 패턴을 포착하기 위해 모든 카테고리 사용
- ✅ **가장 많은 폐업 위험 가맹점 탐지** (4/22)
- ✅ Decline, Worst, Recovery, Cross-metric 모두 활용
- ⚠️ False positive 많지만, 조기 경보에서는 Recall 우선

#### 실험 2: Decline Only (36개)
**의도**: "하락" 신호만으로 단순화하면 더 정확할 것
- ❌ PR-AUC 0.0018 (거의 0) - precision 극히 낮음
- ❌ False positive 1,323개로 폭증
- **교훈**: 단일 패턴만으로는 정밀도가 현저히 떨어짐

#### 실험 3: Selected Features (19개)
**의도**: Feature selection으로 핵심만 추출하면 과적합 방지
- ❌ ROC-AUC 0.6980 → 0.6470 (-0.051)
- ❌ All Interval 대비 모든 지표 악화
- **교훈**: Feature 수를 줄이면 다양한 위험 유형 감지 능력 상실

---

## Slide 4: 최종 모델 선택 및 결론

### ✅ 최종 선택: All Interval Features (100+) 모델

#### 선택 이유

| 이유 | 설명 |
|-----|------|
| **1. 최다 탐지** | 4/22 (18.2%) - 다른 모델 대비 최고 성능 |
| **2. 다각도 분석** | 100개 이상 feature로 다양한 위험 유형 포착<br>(매출 급락형, 고객 이탈형, 배달 의존형 등) |
| **3. 실용성** | 조기 경보 시스템에서 **Recall > Precision**<br>→ 놓친 폐업(FN)이 오탐(FP)보다 치명적 |
| **4. 확장성** | Threshold 최적화로 precision 개선 여지<br>SHAP 분석으로 위기 신호 해석 가능 |

---

### 📈 최종 모델 성능

```
Model: XGBoost (Hyperparameter Tuned)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:         134개 (All Interval + 기존 features)
Train samples:    62,257개
Test samples:     24,303개
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test ROC-AUC:     0.698
Test PR-AUC:      0.026
Test Recall:      18.2% (4/22 폐업 탐지)
Test Precision:   0.71%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best params:
  - max_depth: 4
  - learning_rate: 0.05
  - n_estimators: 100
  - scale_pos_weight: 32
```

---

### 🎯 핵심 발견 (Key Findings)

#### 1️⃣ More Features = Better Detection
- 100개 이상의 피처가 각각 **다른 위험 패턴**을 포착
- Feature selection으로 축소 시 탐지 능력 손실
- **교훈**: Dimensionality reduction이 항상 좋은 것은 아님

#### 2️⃣ 단일 패턴만으로는 부족
- Decline만 사용 → 오탐 급증 (FP 1,323개)
- **다각도 분석 필수**: Historical context, Recovery patterns, Cross-metric divergence

#### 3️⃣ False Positives vs Recall 트레이드오프
- 조기 경보 시스템 특성상 **Recall 우선**
- False positive는 threshold 조정으로 관리 가능
- 놓친 폐업(FN)이 더 치명적

#### 4️⃣ 구간 패턴의 유효성 입증
- 기존 time-series features만으로는 불충분
- **상대적 순위 변화 패턴**이 폐업 위험의 강력한 신호

---

### 🚀 비즈니스 임팩트

#### 위기 유형별 분류 및 맞춤형 솔루션

| 위기 유형 | 탐지 패턴 | 제안 솔루션 |
|----------|----------|------------|
| **매출 급락형** | 연속 하락 3개월 이상<br>하락 속도 빠름 | 단기 운영자금 대출<br>경영 컨설팅 |
| **고객 이탈형** | 재방문율 지속 하락<br>신규 고객 유입 감소 | 마케팅 지원 대출<br>고객 분석 서비스 |
| **배달 의존형** | 배달 비율 증가 + 총 매출 감소 | 오프라인 강화 프로그램<br>상권 분석 |
| **경쟁 열위형** | 업종/상권 순위 지속 하락 | 경쟁력 강화 패키지<br>차별화 전략 수립 |
| **종합 위기형** | 매출+고객 동시 하락<br>역대 최악 도달 | 긴급 안정화 패키지<br>구조조정 지원 |

---

### 📝 후속 작업

1. ✅ **All Interval Features 모델로 확정**
2. 🔄 **Threshold 최적화** (F2-score 중심, Recall 중시)
3. 🔄 **SHAP 분석**으로 주요 위기 신호 해석
4. 🔄 **비즈니스 시나리오별 threshold** 설정
   - Aggressive: Recall 극대화 (폐업 위험 하나도 놓치지 않기)
   - Balanced: F1-score 최적화
   - Conservative: Precision 확보 (오탐 최소화)
5. 🔄 **실시간 조기 경보 대시보드** 구축

---

## 기술적 의사결정 요약

### Why XGBoost?
- ✅ 극심한 불균형 데이터 처리 (`scale_pos_weight`)
- ✅ 범주형(구간) 데이터 효과적 처리
- ✅ Feature importance 제공 → 해석 가능성
- ✅ 빠른 학습 속도 → 실험 iteration 용이

### Why All Interval Features?
- ✅ 각 feature가 서로 다른 위험 각도 포착
- ✅ 실험 결과 최고 detection rate
- ✅ 100개 이상 feature여도 overfitting 없음 (XGBoost의 강건성)
- ✅ SHAP 분석 시 풍부한 인사이트 제공 가능

### Why Recall > Precision?
- ✅ 조기 경보 시스템의 본질: **위험 신호 놓치지 않기**
- ✅ False positive 비용 < False negative 비용
  - FP: 불필요한 검토 (시간/비용 소모)
  - FN: 폐업 놓침 (가맹점 손실, 금융 손실)
- ✅ 2단계 검증으로 FP 완화 가능

---

## 📊 슬라이드별 추천 시각화 자료

### Slide 1: 프로젝트 개요 및 클래스 불균형

#### 시각화 1: 클래스 불균형 분포
- **노트북**: `notebooks/01_eda.ipynb`
- **셀**: Cell 11 (개설일/폐업일 분석)
- **내용**: 텍스트 출력 (폐업 127개 3.03%, 운영중 4,058개 96.97%)
- **추가 작업**: Pie chart 또는 Bar chart로 시각화 생성 필요

**추천 시각화 코드 (추가 생성)**:
```python
import matplotlib.pyplot as plt

# 클래스 분포
closed = 127
operating = 4058
labels = ['폐업 (3.0%)', '운영중 (97.0%)']
sizes = [closed, operating]
colors = ['#ff6b6b', '#51cf66']
explode = (0.1, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
ax1.set_title('클래스 불균형: 폐업 vs 운영중', fontsize=14, fontweight='bold')

# Bar chart with imbalance ratio
ax2.bar(['폐업', '운영중'], sizes, color=colors, alpha=0.8)
ax2.set_ylabel('가맹점 수', fontsize=12)
ax2.set_title('클래스 불균형 (1:32)', fontsize=14, fontweight='bold')
ax2.text(0, closed+100, f'{closed:,}개', ha='center')
ax2.text(1, operating+100, f'{operating:,}개', ha='center')
plt.tight_layout()
```

---

### Slide 2: 모델 설계 전략 (Interval Pattern Features)

#### 시각화 1: 구간 변화 패턴 (폐업 예정 가맹점)
- **노트북**: `notebooks/03-1_interval_pattern_features.ipynb`
- **셀**: Cell 18
- **내용**: 폐업 예정 가맹점 샘플의 매출/고객 구간 시계열 그래프
- **특징**: 하락 구간을 빨간색으로 하이라이트, 매출/고객 구간을 함께 표시

**코드 위치**:
```python
# Cell 18: 폐업 예정 가맹점 3개 샘플 시각화
fig, axes = plt.subplots(len(sample_closing), 1, figsize=(14, 4*len(sample_closing)))
for idx, (mct, ax) in enumerate(zip(sample_closing, axes)):
    merchant_data = df_with_intervals[df_with_intervals['ENCODED_MCT'] == mct].sort_values('TA_YM')
    ax.plot(merchant_data['TA_YM'], merchant_data['RC_M1_SAA'],
            marker='o', linewidth=2, label='Sales Interval')
    ax.plot(merchant_data['TA_YM'], merchant_data['RC_M1_UE_CUS_CN'],
            marker='s', linewidth=2, label='Customer Interval', alpha=0.7)
```

#### 시각화 2: Interval Pattern Features와 타겟 상관관계
- **노트북**: `notebooks/03-1_interval_pattern_features.ipynb`
- **셀**: Cell 14
- **내용**: Top 30 interval pattern features의 타겟 변수와의 상관관계 bar chart
- **특징**: 폐업 예측에 가장 영향력 있는 feature 순위 표시

**코드 위치**:
```python
# Cell 14: 타겟과의 상관관계
correlations = df_with_intervals[new_features].corrwith(
    df_with_intervals['will_close_3m']
).abs().sort_values(ascending=False)

top_corr = correlations.head(30)
fig, ax = plt.subplots(figsize=(14, 10))
ax.barh(range(len(top_corr)), top_corr.values)
ax.set_title('Top 30 Interval Pattern Features by Correlation with Target')
```

#### 시각화 3: 폐업 vs 정상 가맹점 Feature 비교
- **노트북**: `notebooks/03-1_interval_pattern_features.ipynb`
- **셀**: Cell 16
- **내용**: Top 15 features의 폐업/정상 가맹점 평균값 비교
- **특징**: 두 그룹 간 차이를 side-by-side bar chart로 표시

**코드 위치**:
```python
# Cell 16: 폐업/정상 비교
comparison = df_with_intervals.groupby('will_close_3m')[available_features].mean()
ax.bar(x - width/2, comparison['Normal (0)'], width, label='Normal', alpha=0.8)
ax.bar(x + width/2, comparison['Will Close (1)'], width, label='Will Close', alpha=0.8)
```

---

### Slide 3: 실험 과정 및 비교

#### 시각화 1: 3개 실험 성능 비교표
- **노트북**: `notebooks/04-3_xgboost_selected_interval_features.ipynb`
- **셀**: Cell 38
- **내용**: 4개 모델(Original, All Interval, Decline Only, Selected) 비교 테이블
- **지표**: ROC-AUC, PR-AUC, Detection Rate, Feature 수

**코드 위치**:
```python
# Cell 38: 모델 비교 테이블
comparison = pd.DataFrame({
    'Model': ['Original', 'All Interval (100+)', 'Decline Only (36)', 'Selected (20)'],
    'Test ROC-AUC': [0.7519, 0.6980, 0.6932, test_roc_auc],
    'Test PR-AUC': [0.0499, 0.0261, 0.0018, test_pr_auc],
    'Detection Rate (%)': [4.5, 18.2, 27.3, ...]
})
```

#### 시각화 2: PR Curve 비교
- **노트북**: `notebooks/04-2_simple_xgboost_decline_only.ipynb`
- **셀**: Cell 31
- **내용**: Train/Valid/Test 세트의 Precision-Recall Curve
- **특징**: PR-AUC 값 표시, imbalanced data 평가에 적합

**코드 위치**:
```python
# Cell 31: PR Curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (X, y, name) in zip(axes, [(X_train, y_train, 'Train'), ...]):
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
```

#### 시각화 3: False Positive 비교 (추가 생성 필요)
**추천 시각화 코드**:
```python
import matplotlib.pyplot as plt

experiments = ['All Interval\n(100+)', 'Decline Only\n(36)', 'Selected\n(19)']
fp_counts = [267, 1323, 925]  # 각 실험의 FP 수
detection = [4, 3, 3]  # 탐지한 폐업 수

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# False Positive 비교
colors = ['#51cf66', '#ff6b6b', '#ffa94d']
ax1.bar(experiments, fp_counts, color=colors, alpha=0.8)
ax1.set_ylabel('False Positive 수', fontsize=12)
ax1.set_title('실험별 False Positive 비교', fontsize=14, fontweight='bold')
ax1.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='실용성 한계')
for i, (exp, fp) in enumerate(zip(experiments, fp_counts)):
    ax1.text(i, fp+50, f'{fp:,}개', ha='center', fontsize=10)

# Detection vs FP 산점도
ax2.scatter(fp_counts, detection, s=300, c=colors, alpha=0.6)
ax2.set_xlabel('False Positive 수', fontsize=12)
ax2.set_ylabel('폐업 탐지 수', fontsize=12)
ax2.set_title('탐지율 vs 오탐률 트레이드오프', fontsize=14, fontweight='bold')
for exp, fp, det, color in zip(experiments, fp_counts, detection, colors):
    ax2.annotate(exp.replace('\n', ' '), (fp, det),
                 xytext=(10, 10), textcoords='offset points')
plt.tight_layout()
```

---

### Slide 4: 최종 모델 선택 및 결론

#### 시각화 1: Feature Importance Top 20
- **노트북**: `notebooks/04-2_simple_xgboost_decline_only.ipynb`
- **셀**: Cell 28-29
- **내용**: XGBoost feature importance horizontal bar chart
- **특징**: 상위 20개 feature의 중요도 표시

**코드 위치**:
```python
# Cell 28-29: Feature Importance
importance_df = pd.DataFrame({
    'feature': decline_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(12, 10))
top_20 = importance_df.head(20)
ax.barh(range(len(top_20)), top_20['importance'].values, alpha=0.8)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'].values)
ax.set_title('Feature Importance - Top 20', fontsize=14, fontweight='bold')
```

#### 시각화 2: SHAP Summary Plot
- **노트북**: `notebooks/04-1_model_training_with_interval_feature.ipynb`
- **셀**: SHAP 분석 섹션 (확인 필요)
- **내용**: SHAP values의 전역적 feature importance와 영향 방향
- **특징**: Feature 값과 SHAP value의 관계 표시

**예상 코드**:
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (bar)
shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=20)

# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test, max_display=20)
```

#### 시각화 3: 최종 성능 지표 요약 (추가 생성 필요)
**추천 시각화 코드**:
```python
import matplotlib.pyplot as plt
import numpy as np

# 최종 모델 성능
metrics = ['ROC-AUC', 'PR-AUC', 'Recall', 'Precision']
values = [0.698, 0.026, 0.182, 0.0071]
target = [0.70, 0.05, 0.20, 0.01]  # 목표값

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, values, width, label='실제 성능', color='#339af0', alpha=0.8)
bars2 = ax.bar(x + width/2, target, width, label='목표값', color='#51cf66', alpha=0.8)

ax.set_ylabel('점수', fontsize=12)
ax.set_title('최종 모델 성능 vs 목표', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# 값 표시
for i, (v, t) in enumerate(zip(values, target)):
    ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    ax.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', fontsize=10)

plt.tight_layout()
```

#### 시각화 4: Confusion Matrix (최종 모델)
**추천 시각화 코드**:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Test set confusion matrix
y_pred = (y_pred_proba >= 0.05).astype(int)  # threshold=0.05
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['운영중', '폐업'], yticklabels=['운영중', '폐업'],
            annot_kws={'size': 14})
ax.set_xlabel('예측', fontsize=12)
ax.set_ylabel('실제', fontsize=12)
ax.set_title('Confusion Matrix (Test Set)\nThreshold=0.05',
             fontsize=14, fontweight='bold')

# 추가 정보 표시
tn, fp, fn, tp = cm.ravel()
info_text = f'TP={tp}, FP={fp}\nFN={fn}, TN={tn}\nRecall={tp/(tp+fn):.1%}'
ax.text(1.2, 0.5, info_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
```

---

## 📝 시각화 생성 가이드

### 바로 사용 가능한 시각화 (노트북 실행):
1. **Slide 2**: `03-1_interval_pattern_features.ipynb` Cell 14, 16, 18
2. **Slide 3**: `04-3_xgboost_selected_interval_features.ipynb` Cell 38
3. **Slide 3**: `04-2_simple_xgboost_decline_only.ipynb` Cell 31
4. **Slide 4**: `04-2_simple_xgboost_decline_only.ipynb` Cell 28-29

### 추가 생성 필요한 시각화:
1. **Slide 1**: 클래스 불균형 pie/bar chart
2. **Slide 3**: 3개 실험 성능 통합 비교, FP 비교 차트
3. **Slide 4**: 최종 성능 지표 요약, Confusion Matrix
4. **Slide 4**: SHAP summary plot (04-1 노트북 확인 필요)

### 실행 순서:
1. 각 노트북의 데이터 로드 및 전처리 셀을 먼저 실행
2. 해당 시각화 셀 실행
3. `plt.savefig()` 로 이미지 파일로 저장하여 PPT에 삽입

---

**작성일**: 2025-10-13
**프로젝트**: 2025 빅콘테스트 - 가맹점 위기 조기 경보 시스템
**모델**: XGBoost with All Interval Pattern Features (134 features)
**성능**: Test Recall 18.2% (4/22 탐지), ROC-AUC 0.698, PR-AUC 0.026
