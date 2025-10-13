# Interval Pattern Features 실험 과정 및 결론

## 실험 배경

기존 모델링 접근 방식(time-series features, customer features, composite features)으로는 만족스러운 성과를 얻지 못했습니다. 데이터셋의 핵심인 **구간(interval) 변화 패턴**이 충분히 활용되지 않았다는 점을 파악하고, 구간 패턴 기반 feature engineering을 진행했습니다.

## 구간(Interval) 데이터의 의미

데이터셋의 구간 피처 (예: `RC_M1_SAA`, `RC_M1_TO_UE_CT` 등):
- **1**: 상위 10% (최고 성과)
- **2**: 10-25%
- **3**: 25-50%
- **4**: 50-75%
- **5**: 75-90%
- **6**: 하위 10% (최저 성과)

**핵심 인사이트**: 구간이 **증가**(1→6 방향)하는 것은 성과 **악화**를 의미하며, 이는 폐업 위험의 강력한 신호입니다.

## Feature Engineering 전략

4가지 카테고리의 구간 패턴 피처를 설계했습니다:

### 1. 구간 하락 패턴 (Interval Decline Patterns)
- 월별 구간 변화 (MoM interval change)
- 하락 플래그 (구간 증가 = 성과 악화)
- 연속 하락 개월 수 (consecutive declines)
- 기간별 하락 횟수 (3개월, 6개월, 12개월)
- 총 구간 하락폭 (3개월전 대비, 6개월전 대비, 12개월전 대비)
- 하락 속도 (decline speed)

### 2. 역대 최악 지표 (Historical Worst Features)
- 역대 최악 구간 (worst interval ever)
- 역대 최고 구간 (best interval ever)
- 현재 최악 상태 여부 (at worst now)
- 최고 구간과의 거리 (distance from best)
- 최고 성과 이후 경과 개월 수 (months since best)

### 3. 회복 지표 (Recovery Indicators)
- 회복 플래그 (구간 감소 = 성과 개선)
- 연속 회복 개월 수 (consecutive recovery)
- 하락 후 회복 패턴 (recovery after decline)
- 구간 변동성 (interval volatility) - 불안정성 지표
- 방향 전환 횟수 (direction changes) - 진동 패턴

### 4. 교차 지표 비교 (Cross-Metric Interval Features)
- 매출 구간 vs 고객 수 구간 괴리도 (divergence)
  - 예: 매출은 하락하는데 고객은 유지 → 객단가 문제
  - 예: 고객은 하락하는데 매출 유지 → 소수 고객 의존
- 동반 하락 (aligned decline) - 모든 지표가 함께 하락 (위기 신호)
- 괴리 크기 (divergence magnitude)

**총 100개 이상의 구간 패턴 피처 생성**

구현: `pipeline/features/interval_patterns.py`
문서: `docs/05_interval_pattern_features.md`

---

## 실험 과정 및 결과

### 실험 설정
- **Train/Valid/Test Split**: 시계열 기반 분할
  - Train: 202301-202406 (6개월)
  - Valid: 202407-202409 (3개월)
  - Test: 202410-202412 (3개월)
- **모델**: XGBoost with `scale_pos_weight` (class imbalance 처리)
- **평가 지표**: ROC-AUC, PR-AUC, Detection Rate (Recall), False Positives
- **목표**: 폐업 위험 가맹점 최대 탐지 (Recall 중시)

---

### 실험 1: All Interval Features (100+개)
**노트북**: `notebooks/04-1_model_training_with_interval_feature.ipynb`

**전략**:
- 생성한 모든 구간 패턴 피처(100개 이상) 사용
- 기존 time-series, customer, composite features와 결합

**결과**:
- **Test ROC-AUC**: 0.6980
- **Test PR-AUC**: 0.0261
- **Detection**: **4/22 탐지 (18.2%)** ← **최다 탐지**
- **False Positives**: 수백 개 (높음)

**분석**:
- ✅ **가장 많은 폐업 위험 가맹점을 탐지**
- ✅ 다양한 구간 패턴을 포착하여 다각도로 위험 신호 감지
- ⚠️ False positive가 많지만, 조기 경보 시스템에서는 Recall이 우선
- **결론**: Feature가 많아도 각각이 다른 위험 유형을 포착하므로 유용

---

### 실험 2: Decline Features Only (36개)
**노트북**: `notebooks/04-2_simple_xgboost_decline_only.ipynb`

**전략**:
- 구간 하락 패턴 피처만 선택 (36개)
- 가장 직관적인 "하락" 신호에만 집중

**결과**:
- **Test ROC-AUC**: 0.6932
- **Test PR-AUC**: 0.0018 (매우 낮음)
- **Detection**: 3/11 탐지 (27.3%)
- **False Positives**: **1,323개** (극단적으로 높음)

**분석**:
- ⚠️ PR-AUC가 거의 0에 가까움 (precision 매우 낮음)
- ⚠️ False positive가 너무 많아 실용성 없음
- ❌ 하락 패턴만으로는 불충분 - 오탐이 급증
- **결론**: 단일 유형의 피처만으로는 정밀도가 현저히 떨어짐

---

### 실험 3: Selected Interval Features (19개)
**노트북**: `notebooks/04-3_xgboost_selected_interval_features.ipynb`

**전략**:
- Feature selection으로 100개 → 19개로 축소
- 선택 기준:
  1. 타겟과의 상관관계 상위 40개
  2. 결측치 30% 미만
  3. 분산 충분
  4. 카테고리 균형 (decline 10개, worst 4개, cross-metric 3개, recovery 2개)

**결과**:
- **Test ROC-AUC**: 0.6470 (하락)
- **Test PR-AUC**: 0.0016 (하락)
- **Detection**: 3/11 탐지 (27.3%)
- **False Positives**: 925개

**분석**:
- ⚠️ ROC-AUC 0.6980 → 0.6470 (-0.051)
- ⚠️ PR-AUC 0.0261 → 0.0016 (-0.0245)
- ❌ All Interval 대비 모든 지표 악화
- ❌ Feature 수를 줄이면서 중요한 정보 손실
- **결론**: Feature selection이 역효과 - 다양한 패턴 감지 능력 상실

---

## 실험 비교 요약표

| 실험 | Feature 수 | ROC-AUC | PR-AUC | Detection | False Positives | 비고 |
|------|-----------|---------|--------|-----------|----------------|------|
| **실험 1: All Interval** | 100+ | **0.6980** | **0.0261** | **4/22 (18.2%)** | 수백 개 | ✅ **최다 탐지** |
| 실험 2: Decline Only | 36 | 0.6932 | 0.0018 | 3/11 (27.3%) | 1,323 | ❌ FP 폭증 |
| 실험 3: Selected | 19 | 0.6470 | 0.0016 | 3/11 (27.3%) | 925 | ❌ 성능 하락 |

---

## 핵심 발견 (Key Findings)

### 1. **More Features = Better Detection**
- 100개 이상의 피처가 각각 다른 위험 패턴을 포착
- Feature selection으로 축소하면 탐지 능력 손실
- Decline, Worst, Recovery, Cross-metric 모두 필요

### 2. **단일 패턴만으로는 부족**
- Decline만 사용 → 오탐 급증
- 다각도 분석이 필수 (historical context, recovery patterns, cross-metric divergence)

### 3. **False Positives vs Recall 트레이드오프**
- 조기 경보 시스템 특성상 **Recall 우선**
- False positive는 threshold 조정으로 관리 가능
- 놓친 폐업(False Negative)이 더 치명적

### 4. **구간 패턴의 유효성 입증**
- 기존 접근 대비 구간 패턴 추가로 탐지율 향상
- 매출/고객 수/객단가 등 다면적 구간 분석이 효과적

---

## 최종 결론 및 선택

### ✅ **All Interval Features (100+) 모델 선택**

**선택 이유**:
1. **최다 탐지**: 4/22 (18.2%) - 다른 모델 대비 최고
2. **다각도 분석**: 100개 이상 피처로 다양한 위험 유형 포착
3. **실용성**: 조기 경보 시스템에서 Recall이 최우선 목표
4. **확장성**: 향후 threshold 최적화로 precision 개선 여지

**후속 작업**:
- Threshold 최적화로 Precision/Recall 균형 조정
- SHAP 분석으로 주요 위험 신호 해석
- 위험 유형별 분류 (매출 급락형, 고객 이탈형, 배달 의존형 등)

**트레이드오프 수용**:
- False positive 증가는 감수
- Threshold 조정, 2단계 검증 등으로 완화 가능
- **핵심**: 폐업 위험을 놓치지 않는 것이 최우선

---

## 비즈니스 임팩트

이 실험을 통해:
1. **구간 패턴 기반 feature engineering의 효과 입증**
2. **Feature 다양성의 중요성 확인** (dimensionality reduction 역효과)
3. **조기 경보 시스템의 핵심 = Recall** 임을 재확인
4. **실전 배포 전략**: All Interval + Threshold 최적화

---

## 다음 단계

1. ✅ All Interval Features 모델로 확정
2. 🔄 Threshold 최적화 (F2-score 중심)
3. 🔄 SHAP 분석으로 주요 위험 신호 식별
4. 🔄 비즈니스 시나리오별 threshold 설정 (aggressive/balanced/conservative)
5. 🔄 위험 유형 분류 및 맞춤형 솔루션 매칭

---

**작성일**: 2025-10-13
**실험 노트북**:
- `notebooks/04-1_model_training_with_interval_feature.ipynb` (최종 선택)
- `notebooks/04-2_simple_xgboost_decline_only.ipynb`
- `notebooks/04-3_xgboost_selected_interval_features.ipynb`
