# 앙상블 모델 구현 계획

## 목표
04_model_training.ipynb에서 학습된 XGBoost와 LightGBM 모델을 활용하여 앙상블 모델을 구현하고 성능을 향상시킨다.

## 배경
- 현재 상황: XGBoost (Tuned) 단일 모델이 최고 성능 (Test ROC-AUC: 0.9260, F1: 0.5034)
- 문제점: 앙상블 모델 구현이 원래 계획에 있었으나 04_model_training.ipynb에서 누락됨
- 기회: 이미 `pipeline/models/ensemble.py`에 앙상블 클래스가 구현되어 있어 바로 활용 가능

## 노트북 파일
- **파일명**: `notebooks/041_ensemble_model_training.ipynb`
- **목적**: 다양한 앙상블 기법을 비교하고 최적의 앙상블 모델 선정

---

## 구현 내용

### 1. 환경 설정
- 필요한 라이브러리 임포트
- 커스텀 모듈 임포트 (`pipeline.models`, `pipeline.evaluation`)
- 데이터 및 기존 모델 로드

**데이터**:
- `data/processed/featured_data.csv` 로드
- Train/Val/Test 분할 (04_model_training.ipynb와 동일하게 시계열 고려)
  - Train: 2023년 (202301-202312)
  - Validation: 2024년 1-6월 (202401-202406)
  - Test: 2024년 7-12월 (202407-202412)

**모델 로드**:
- `models/xgboost_best.pkl`: Hyperparameter tuning된 XGBoost
- `models/lightgbm_baseline.pkl`: LightGBM baseline
- `models/feature_cols.pkl`: Feature 목록

### 2. 앙상블 방법론

#### 2.1 Simple Averaging Ensemble
- **방법**: XGBoost와 LightGBM의 예측 확률을 단순 평균
- **구현**: `(pred_xgb + pred_lgb) / 2`
- **특징**: 가장 간단한 앙상블, 베이스라인으로 사용

#### 2.2 Weighted Ensemble (성능 기반)
- **방법**: Validation 성능에 비례하는 가중치 사용
- **가중치 계산**:
  - w_xgb = ROC_AUC_xgb / (ROC_AUC_xgb + ROC_AUC_lgb)
  - w_lgb = ROC_AUC_lgb / (ROC_AUC_xgb + ROC_AUC_lgb)
- **구현**: `EnsembleModel(models=[xgb, lgb], weights=[w_xgb, w_lgb])`

#### 2.3 Optimized Weighted Ensemble
- **방법**: `EnsembleModel.optimize_weights()` 사용
- **최적화 목표**: Validation ROC-AUC 최대화
- **알고리즘**: Scipy SLSQP 최적화
- **특징**: 데이터 기반으로 최적 가중치 자동 탐색

#### 2.4 Voting Ensemble
- **Soft Voting**: 예측 확률 평균 후 threshold 적용
- **Hard Voting**: 각 모델의 예측 결과를 다수결로 결정
- **구현**: `VotingEnsemble(models=[xgb, lgb], voting='soft'/'hard')`

#### 2.5 Stacking Ensemble (선택적)
- **방법**:
  1. Base models (XGBoost, LightGBM)의 예측 확률을 feature로 사용
  2. Meta model (Logistic Regression)로 최종 예측
- **구현**:
  - Train set에 대해 out-of-fold 예측 생성
  - Meta model 학습
- **장점**: 모델 간 상호작용 학습 가능

### 3. 성능 평가 및 비교

#### 3.1 평가 메트릭
- **ROC-AUC**: 전체적인 분류 성능
- **PR-AUC**: 불균형 데이터에 특화된 메트릭
- **F1 Score**: Precision과 Recall의 조화 평균
- **Precision**: 폐업 예측의 정확도
- **Recall**: 실제 폐업의 탐지율

#### 3.2 비교 대상
1. XGBoost (Baseline)
2. XGBoost (Tuned) - 현재 최고 성능
3. LightGBM (Baseline)
4. Simple Averaging Ensemble
5. Weighted Ensemble (성능 기반)
6. Optimized Weighted Ensemble
7. Soft Voting Ensemble
8. Hard Voting Ensemble
9. Stacking Ensemble (선택적)

#### 3.3 비교 테이블 생성
- Validation/Test 성능을 한눈에 비교하는 DataFrame 생성
- 모든 메트릭 포함 (ROC-AUC, PR-AUC, F1, Precision, Recall)

#### 3.4 시각화
- **ROC Curves**: 모든 모델을 한 그래프에 표시
- **PR Curves**: 모든 모델을 한 그래프에 표시
- **Confusion Matrices**: 최고 성능 모델들 비교
- **Performance Bar Chart**: 메트릭별 모델 성능 비교

### 4. 앙상블 분석

#### 4.1 가중치 분석
- Optimized Ensemble의 최적 가중치 확인
- 가중치가 모델 성능과 어떻게 연관되는지 분석

#### 4.2 모델 간 예측 차이 분석
- XGBoost와 LightGBM의 예측이 크게 다른 케이스 탐색
- 두 모델이 동의하는 케이스 vs 불일치하는 케이스

#### 4.3 Hard Cases 분석
- 모든 모델이 틀리는 어려운 케이스 분석
- 어떤 feature가 이런 케이스에서 문제인지 파악

### 5. 최종 모델 선정 및 저장

#### 5.1 모델 선정 기준
- **Primary**: Test ROC-AUC (일반화 성능)
- **Secondary**: Test F1 Score (균형 잡힌 성능)
- **고려사항**: Precision-Recall trade-off

#### 5.2 저장할 파일
1. **모델 파일**:
   - `models/ensemble_best.pkl`: 최고 성능 앙상블 모델
   - `models/ensemble_simple_avg.pkl`: Simple averaging (백업)
   - `models/ensemble_optimized.pkl`: Optimized weights (백업)

2. **메타데이터**:
   - `models/ensemble_results.json`:
     ```json
     {
       "best_model": "Optimized Weighted Ensemble",
       "test_roc_auc": 0.xxxx,
       "test_f1": 0.xxxx,
       "weights": {
         "xgboost": 0.xx,
         "lightgbm": 0.xx
       },
       "comparison_table": {...}
     }
     ```

3. **비교 결과**:
   - `models/ensemble_comparison.csv`: 모든 모델 성능 비교표

### 6. 결론 및 다음 단계

#### 6.1 예상 결과
- 앙상블 모델이 단일 모델보다 0.5-2% 성능 향상 예상
- Voting보다 Weighted Ensemble이 더 나은 성능 기대
- Stacking은 복잡도 대비 성능 향상이 크지 않을 수 있음

#### 6.2 다음 단계
1. **임계값 최적화**: Business metric 기반 threshold 조정
2. **Feature Re-engineering**: 앙상블 분석 결과를 바탕으로 feature 개선
3. **모델 해석**: SHAP value를 앙상블 모델에 적용
4. **프로덕션 준비**:
   - 모델 서빙 파이프라인 구축
   - 모니터링 시스템 설계
   - A/B 테스트 계획

---

## 코드 구조

### 활용할 기존 모듈
```python
from pipeline.models import EnsembleModel, VotingEnsemble
from pipeline.evaluation import ModelEvaluator, calculate_metrics
from pipeline.evaluation import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
```

### 노트북 섹션 구성
1. 환경 설정 (1-2 cells)
2. 데이터 및 모델 로드 (3-5 cells)
3. Simple & Weighted Ensemble (6-10 cells)
4. Voting Ensemble (11-13 cells)
5. Optimized Ensemble (14-16 cells)
6. Stacking Ensemble - 선택적 (17-20 cells)
7. 성능 비교 및 시각화 (21-25 cells)
8. 앙상블 분석 (26-30 cells)
9. 최종 모델 저장 (31-33 cells)
10. 결론 (34 cell)

---

## 주의사항

1. **시계열 특성 유지**: Train/Val/Test 분할 시 시간 순서 유지 필수
2. **데이터 일관성**: 04_model_training.ipynb와 동일한 전처리 적용
3. **Overfitting 방지**: Validation set에만 최적화, Test는 최종 평가용
4. **재현성**: Random seed 고정 (42)
5. **메모리 관리**: SHAP 계산 시 샘플링 사용

## 성공 기준

- [ ] 최소 3가지 이상의 앙상블 방법 구현 및 비교
- [ ] Test ROC-AUC가 기존 XGBoost (0.9260)보다 향상
- [ ] 모든 앙상블 모델의 성능 비교표 생성
- [ ] 최고 성능 앙상블 모델 저장 완료
- [ ] 앙상블 가중치 및 모델 간 차이 분석 완료
