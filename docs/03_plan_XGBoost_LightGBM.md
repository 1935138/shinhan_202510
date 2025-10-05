# XGBoost/LightGBM 기반 가맹점 위기 조기 경보 시스템 구현 계획

## 📋 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [구현 아키텍처](#2-구현-아키텍처)
3. [단계별 구현 계획](#3-단계별-구현-계획)
4. [기술 스택](#4-기술-스택)
5. [구현 세부사항](#5-구현-세부사항)
6. [디렉토리 구조](#6-디렉토리-구조)
7. [예상 결과물](#7-예상-결과물)

---

## 1. 프로젝트 개요

### 1.1 목표
**영세/중소 요식 가맹점의 경영 위기를 1~3개월 전에 예측하는 AI 조기 경보 시스템 개발**

### 1.2 핵심 과제
- 폐업 위험 가맹점 조기 식별 (Recall 최대화)
- 위기 신호 해석 및 인사이트 도출 (SHAP)
- 맞춤형 금융상품/서비스 제안

### 1.3 왜 XGBoost/LightGBM인가?

#### 선택 이유
1. **불균형 데이터 처리 우수**: 폐업률 3% vs 운영중 97%
2. **해석 가능성**: Feature Importance + SHAP values
3. **결측값 자동 처리**: SV(-999999.9) 값 포함
4. **범주형 데이터 효과적 처리**: 6단계 구간 데이터
5. **빠른 학습 속도**: 다양한 실험 가능
6. **검증된 성능**: Kaggle 등 대회 상위권 솔루션

#### 장점 비교

| 특성 | XGBoost | LightGBM |
|------|---------|----------|
| **학습 속도** | 보통 | 빠름 ⚡ |
| **메모리 효율** | 보통 | 우수 |
| **대용량 데이터** | 제한적 | 우수 |
| **불균형 처리** | `scale_pos_weight` | `is_unbalance=True` |
| **범주형 처리** | Label encoding 필요 | 자동 지원 |
| **과적합 방지** | 우수 | 우수 |

**→ 두 모델 모두 학습 후 앙상블로 최종 성능 극대화**

---

## 2. 구현 아키텍처

### 2.1 전체 파이프라인

```
[원본 데이터]
    ↓
[데이터 전처리]
- 날짜 변환
- SV 처리
- 구간 인코딩
    ↓
[Feature Engineering]
- 시계열 특성 (lag, rolling)
- 상대적 지표 (순위 변화)
- 고객 행동 (재방문율 추세)
- 복합 지표 (매출-고객 괴리)
    ↓
[Train/Valid/Test Split]
- 시간 기반 분할
- 202401-202406: Train
- 202407-202409: Valid
- 202410-202412: Test
    ↓
[불균형 처리]
- SMOTE (Over-sampling)
- Class Weight 조정
    ↓
[모델 학습]
- XGBoost
- LightGBM
- (CatBoost - 옵션)
    ↓
[앙상블]
- Voting (Soft)
- Stacking
    ↓
[모델 해석]
- SHAP values
- Feature Importance
- Partial Dependence Plot
    ↓
[위기 신호 분석]
- 주요 위험 요인 Top 10
- 업종/상권별 패턴
- 위험 유형 분류
    ↓
[비즈니스 제안]
- 금융상품 매칭
- 개입 전략
```

### 2.2 모델 구조

#### 개별 모델
```python
# XGBoost 모델
xgb_model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=32,  # (4058/127)
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)

# LightGBM 모델
lgb_model = LGBMClassifier(
    objective='binary',
    is_unbalance=True,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```

#### 앙상블 전략
1. **Soft Voting**: 확률 평균
2. **Stacking**: Logistic Regression meta-learner
3. **가중 평균**: 검증 성능 기반 가중치

---

## 3. 단계별 구현 계획

### Week 1: 데이터 전처리 및 기본 Feature (11/18 ~ 11/24)

#### 목표
- 3개 데이터셋 통합
- 결측값(SV) 처리
- 기본 타겟 변수 생성

#### 작업 내역
1. **데이터 로드 및 병합**
   ```python
   # Dataset 1 + 2 + 3 통합
   # Step 1: Dataset 2(월별 매출/이용 현황) + Dataset 3(월별 고객 정보)를 가맹점ID와 년월 기준으로 병합
   #         → 같은 가맹점의 같은 월 데이터를 옆으로 붙임
   df_merged = df2.merge(df3, on=['ENCODED_MCT', 'TA_YM'])

   # Step 2: 위 결과 + Dataset 1(가맹점 기본정보: 업종, 위치, 폐업일)을 가맹점ID 기준으로 병합
   #         → 가맹점별 고정 정보를 모든 월 데이터에 추가
   df_full = df_merged.merge(df1[['ENCODED_MCT', ...]], on='ENCODED_MCT')

   # 최종: 각 행이 "특정 가맹점의 특정 월" 데이터가 됨 (시계열 분석 가능)
   ```

2. **타겟 변수 생성**
   ```python
   # 폐업 여부
   df['is_closed'] = df['MCT_ME_D'].notna().astype(int)

   # 미래 N개월 내 폐업 (조기 경보용)
   df['will_close_1m'] = ...
   df['will_close_3m'] = ...
   ```

3. **결측값 처리**
   ```python
   # SV(-999999.9) → NaN 변환
   df.replace(-999999.9, np.nan, inplace=True)

   # 전략:
   # - 배달매출: 0으로 대체 (배달 미운영)
   # - 고객정보: 중앙값 또는 별도 플래그 생성
   ```

4. **구간 데이터 인코딩**
   ```python
   interval_mapping = {
       '1_10%이하': 1,
       '2_10-25%': 2,
       '3_25-50%': 3,
       '4_50-75%': 4,
       '5_75-90%': 5,
       '6_90%초과(하위 10% 이하)': 6
   }
   ```

#### 산출물
- `notebooks/02_preprocessing.ipynb`
- `pipeline/preprocessing/data_loader.py`
- `pipeline/preprocessing/feature_encoder.py`

---

### Week 2: 시계열 Feature Engineering (11/25 ~ 12/01)

#### 목표
- 시계열 특성 생성 (100+ features)
- Feature selection

#### 작업 내역

1. **시계열 기본 특성**
   ```python
   # Lag features
   for col in ['RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN']:
       for lag in [1, 3, 6, 12]:
           df[f'{col}_lag_{lag}'] = df.groupby('ENCODED_MCT')[col].shift(lag)

   # 이동평균
   for window in [3, 6]:
       df[f'sales_ma_{window}m'] = df.groupby('ENCODED_MCT')['sales'].transform(
           lambda x: x.rolling(window, min_periods=1).mean()
       )

   # 증감률
   df['sales_mom'] = df.groupby('ENCODED_MCT')['sales'].pct_change()
   df['sales_qoq'] = df.groupby('ENCODED_MCT')['sales'].pct_change(3)
   ```

2. **추세 및 변동성**
   ```python
   # 연속 하락 개월 수
   df['consecutive_decline'] = ...

   # 변동 계수
   df['cv_sales_3m'] = (
       df.groupby('ENCODED_MCT')['sales'].transform(lambda x: x.rolling(3).std()) /
       df.groupby('ENCODED_MCT')['sales'].transform(lambda x: x.rolling(3).mean())
   )
   ```

3. **상대적 지표**
   ```python
   # 업종 내 순위 변화
   df['rank_change_industry'] = df.groupby('ENCODED_MCT')['M12_SME_RY_SAA_PCE_RT'].diff()

   # 업종 평균 대비 격차
   df['gap_from_industry'] = df['M1_SME_RY_SAA_RAT'] - 100
   ```

4. **고객 행동 지표**
   ```python
   # 재방문율 변화
   df['revisit_rate_change'] = df.groupby('ENCODED_MCT')['MCT_UE_CLN_REU_RAT'].diff()

   # 고객 다양성 (Entropy)
   from scipy.stats import entropy
   age_cols = ['M12_MAL_1020_RAT', 'M12_MAL_30_RAT', ...]
   df['customer_diversity'] = df[age_cols].apply(
       lambda row: entropy(row / row.sum()) if row.sum() > 0 else 0, axis=1
   )
   ```

5. **복합 지표**
   ```python
   # 매출-고객 괴리도
   df['sales_customer_gap'] = (
       df['sales_change'] - df['customer_change']
   )

   # 배달 의존도 증가 + 매출 감소
   df['delivery_risk'] = (
       (df['delivery_ratio_change'] > 10) & (df['sales_change'] < 0)
   ).astype(int)
   ```

6. **Feature Selection**
   ```python
   from sklearn.feature_selection import mutual_info_classif

   # 상호정보량 기반 선택
   mi_scores = mutual_info_classif(X, y, random_state=42)
   top_features = mi_scores.argsort()[-100:]  # 상위 100개
   ```

#### 산출물
- `notebooks/03_feature_engineering.ipynb`
- `pipeline/features/time_series_features.py`
- `pipeline/features/customer_features.py`
- `pipeline/features/composite_features.py`

---

### Week 3: 모델 개발 (XGBoost/LightGBM) (12/02 ~ 12/08)

#### 목표
- XGBoost/LightGBM 베이스라인 구축
- 불균형 데이터 처리
- 초기 성능 평가

#### 작업 내역

1. **데이터 분할**
   ```python
   # 시간 기반 분할
   train = df[df['TA_YM'] <= '202406']
   valid = df[(df['TA_YM'] > '202406') & (df['TA_YM'] <= '202409')]
   test = df[df['TA_YM'] > '202409']

   # 특성/타겟 분리
   feature_cols = [col for col in train.columns if col not in
                   ['ENCODED_MCT', 'TA_YM', 'is_closed', 'MCT_ME_D', ...]]
   X_train, y_train = train[feature_cols], train['is_closed']
   ```

2. **불균형 처리**
   ```python
   # SMOTE Over-sampling
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(sampling_strategy=0.3, random_state=42)
   X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

   # Class Weight 계산
   scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
   ```

3. **XGBoost 학습**
   ```python
   import xgboost as xgb

   xgb_model = xgb.XGBClassifier(
       objective='binary:logistic',
       eval_metric='aucpr',
       scale_pos_weight=scale_pos_weight,
       max_depth=6,
       learning_rate=0.05,
       n_estimators=1000,
       subsample=0.8,
       colsample_bytree=0.8,
       gamma=1,
       random_state=42
   )

   xgb_model.fit(
       X_train, y_train,
       eval_set=[(X_valid, y_valid)],
       early_stopping_rounds=50,
       verbose=50
   )
   ```

4. **LightGBM 학습**
   ```python
   import lightgbm as lgb

   lgb_model = lgb.LGBMClassifier(
       objective='binary',
       metric='auc',
       is_unbalance=True,
       max_depth=6,
       learning_rate=0.05,
       n_estimators=1000,
       subsample=0.8,
       colsample_bytree=0.8,
       min_child_samples=20,
       random_state=42
   )

   lgb_model.fit(
       X_train, y_train,
       eval_set=[(X_valid, y_valid)],
       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
   )
   ```

5. **평가**
   ```python
   from sklearn.metrics import classification_report, precision_recall_curve, auc

   # 예측
   y_pred_xgb = xgb_model.predict(X_valid)
   y_proba_xgb = xgb_model.predict_proba(X_valid)[:, 1]

   # PR-AUC
   precision, recall, _ = precision_recall_curve(y_valid, y_proba_xgb)
   pr_auc = auc(recall, precision)

   # Classification Report
   print(classification_report(y_valid, y_pred_xgb))
   ```

#### 산출물
- `notebooks/04_model_xgboost.ipynb`
- `notebooks/05_model_lightgbm.ipynb`
- `pipeline/models/xgboost_model.py`
- `pipeline/models/lightgbm_model.py`
- `models/xgb_baseline.pkl`
- `models/lgb_baseline.pkl`

---

### Week 4: 하이퍼파라미터 튜닝 및 앙상블 (12/09 ~ 12/15)

#### 목표
- Grid Search / Optuna 튜닝
- 앙상블 모델 구축
- 최종 성능 최적화

#### 작업 내역

1. **하이퍼파라미터 튜닝 (Optuna)**
   ```python
   import optuna

   def objective(trial):
       params = {
           'max_depth': trial.suggest_int('max_depth', 3, 10),
           'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
           'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
           'subsample': trial.suggest_float('subsample', 0.6, 1.0),
           'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
           'gamma': trial.suggest_float('gamma', 0, 5),
           'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
           'scale_pos_weight': scale_pos_weight
       }

       model = xgb.XGBClassifier(**params, random_state=42)
       model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 early_stopping_rounds=50,
                 verbose=False)

       y_proba = model.predict_proba(X_valid)[:, 1]
       precision, recall, _ = precision_recall_curve(y_valid, y_proba)
       pr_auc = auc(recall, precision)

       return pr_auc

   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   best_params = study.best_params
   ```

2. **시계열 교차 검증**
   ```python
   from sklearn.model_selection import TimeSeriesSplit

   tscv = TimeSeriesSplit(n_splits=5, test_size=3)
   pr_aucs = []

   for train_idx, val_idx in tscv.split(X):
       X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
       y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

       model.fit(X_tr, y_tr)
       y_proba = model.predict_proba(X_val)[:, 1]

       precision, recall, _ = precision_recall_curve(y_val, y_proba)
       pr_aucs.append(auc(recall, precision))

   print(f"Average PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
   ```

3. **앙상블 - Voting Classifier**
   ```python
   from sklearn.ensemble import VotingClassifier

   voting_clf = VotingClassifier(
       estimators=[
           ('xgb', xgb_model),
           ('lgb', lgb_model)
       ],
       voting='soft',
       weights=[0.6, 0.4]  # 검증 성능 기반
   )

   voting_clf.fit(X_train, y_train)
   ```

4. **앙상블 - Stacking**
   ```python
   from sklearn.ensemble import StackingClassifier
   from sklearn.linear_model import LogisticRegression

   stacking_clf = StackingClassifier(
       estimators=[
           ('xgb', xgb_model),
           ('lgb', lgb_model)
       ],
       final_estimator=LogisticRegression(class_weight='balanced'),
       cv=TimeSeriesSplit(n_splits=5)
   )

   stacking_clf.fit(X_train, y_train)
   ```

5. **임계값 최적화**
   ```python
   # Precision-Recall 트레이드오프
   precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)

   # F2-score 최대화 (Recall 중시)
   from sklearn.metrics import fbeta_score
   f2_scores = [fbeta_score(y_valid, y_proba >= t, beta=2) for t in thresholds]
   optimal_threshold = thresholds[np.argmax(f2_scores)]

   # 최적 임계값 적용
   y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
   ```

#### 산출물
- `notebooks/06_hyperparameter_tuning.ipynb`
- `notebooks/07_ensemble.ipynb`
- `pipeline/models/ensemble.py`
- `models/xgb_tuned.pkl`
- `models/lgb_tuned.pkl`
- `models/ensemble_voting.pkl`
- `models/ensemble_stacking.pkl`

---

### Week 5: 모델 해석 (SHAP) 및 인사이트 도출 (12/16 ~ 12/22)

#### 목표
- SHAP values 분석
- 위기 신호 Top 10 도출
- 업종/상권별 패턴 분석

#### 작업 내역

1. **SHAP 분석 - 전역적 해석**
   ```python
   import shap

   # SHAP Explainer
   explainer = shap.TreeExplainer(xgb_model)
   shap_values = explainer.shap_values(X_test)

   # Summary Plot (전체 특성 중요도)
   shap.summary_plot(shap_values, X_test, plot_type='bar')
   shap.summary_plot(shap_values, X_test)

   # Feature Importance
   feature_importance = pd.DataFrame({
       'feature': X_test.columns,
       'shap_importance': np.abs(shap_values).mean(axis=0)
   }).sort_values('shap_importance', ascending=False)

   print("위기 신호 Top 10:")
   print(feature_importance.head(10))
   ```

2. **SHAP 분석 - 개별 가맹점 해석**
   ```python
   # 폐업 예측된 가맹점 사례
   high_risk_idx = np.where(y_proba_test > 0.8)[0][0]

   # Waterfall Plot
   shap.waterfall_plot(shap.Explanation(
       values=shap_values[high_risk_idx],
       base_values=explainer.expected_value,
       data=X_test.iloc[high_risk_idx],
       feature_names=X_test.columns.tolist()
   ))

   # Force Plot
   shap.force_plot(
       explainer.expected_value,
       shap_values[high_risk_idx],
       X_test.iloc[high_risk_idx]
   )
   ```

3. **Dependence Plot (변수 간 상호작용)**
   ```python
   # 매출 대비 재방문율 상호작용
   shap.dependence_plot(
       'M1_SME_RY_SAA_RAT',
       shap_values,
       X_test,
       interaction_index='MCT_UE_CLN_REU_RAT'
   )
   ```

4. **업종별 위험 패턴 분석**
   ```python
   # 업종별 주요 위험 요인
   for industry in df['HPSN_MCT_ZCD_NM'].unique():
       industry_data = df[df['HPSN_MCT_ZCD_NM'] == industry]
       industry_idx = industry_data.index

       industry_shap = shap_values[industry_idx]
       top_features = np.abs(industry_shap).mean(axis=0).argsort()[-5:]

       print(f"\n{industry} 업종 위험 요인:")
       print(X_test.columns[top_features])
   ```

5. **위험 유형 분류**
   ```python
   def classify_risk_type(shap_row, feature_names):
       top_risk = feature_names[np.abs(shap_row).argsort()[-3:]]

       if 'sales_' in ' '.join(top_risk):
           return '매출 급락형'
       elif 'customer_' in ' '.join(top_risk) or 'revisit' in ' '.join(top_risk):
           return '고객 이탈형'
       elif 'delivery' in ' '.join(top_risk):
           return '배달 의존형'
       elif 'rank_' in ' '.join(top_risk):
           return '경쟁 열위형'
       else:
           return '종합 위기형'

   df_test['risk_type'] = [
       classify_risk_type(shap_values[i], X_test.columns)
       for i in range(len(shap_values))
   ]

   print(df_test['risk_type'].value_counts())
   ```

6. **Partial Dependence Plot**
   ```python
   from sklearn.inspection import partial_dependence, PartialDependenceDisplay

   features = ['M1_SME_RY_SAA_RAT', 'MCT_UE_CLN_REU_RAT', 'rank_change_industry']

   fig, ax = plt.subplots(figsize=(14, 4))
   PartialDependenceDisplay.from_estimator(
       xgb_model, X_test, features, ax=ax
   )
   plt.tight_layout()
   plt.show()
   ```

#### 산출물
- `notebooks/08_shap_analysis.ipynb`
- `notebooks/09_insights.ipynb`
- `pipeline/visualization/shap_plots.py`
- `results/feature_importance.csv`
- `results/risk_patterns_by_industry.csv`
- `results/shap_summary.png`
- `results/risk_type_classification.csv`

---

### Week 6: 비즈니스 제안 및 결과물 정리 (12/23 ~ 12/29)

#### 목표
- 금융상품 매칭
- 최종 결과물 정리
- 발표 자료 준비

#### 작업 내역

1. **위험 유형별 금융상품 매칭**
   ```python
   # 위험 유형별 맞춤 솔루션
   risk_solution_map = {
       '매출 급락형': {
           'product': '마케팅 지원 대출',
           'strategy': [
               '온라인 마케팅 비용 지원',
               '배달 플랫폼 입점 지원',
               '프로모션 운영 자금'
           ],
           'expected_effect': '신규 고객 유입 증가, 매출 15% 회복'
       },
       '고객 이탈형': {
           'product': '고객 리텐션 컨설팅 + CRM 시스템',
           'strategy': [
               '재방문 고객 포인트 적립',
               '단골 고객 전용 혜택',
               'VIP 고객 관리 프로그램'
           ],
           'expected_effect': '재방문율 20% 향상'
       },
       '배달 의존형': {
           'product': '오프라인 강화 프로그램',
           'strategy': [
               '매장 리뉴얼 지원 대출',
               '홀 서비스 개선 컨설팅',
               '포장 용기 개선 지원'
           ],
           'expected_effect': '오프라인 매출 비중 30% 증가'
       },
       '경쟁 열위형': {
           'product': '경쟁력 강화 패키지',
           'strategy': [
               '메뉴 차별화 컨설팅',
               '가격 경쟁력 분석',
               '원가 절감 방안 제시'
           ],
           'expected_effect': '업종 내 순위 20% 상승'
       },
       '종합 위기형': {
           'product': '경영 안정화 특별 패키지',
           'strategy': [
               '긴급 운영 자금 지원',
               '전문가 1:1 컨설팅',
               '업종 전환 지원'
           ],
           'expected_effect': '폐업 위험 50% 감소'
       }
   }

   # 가맹점별 추천 상품
   df_recommendation = df_test[df_test['y_pred'] == 1].copy()
   df_recommendation['recommended_product'] = df_recommendation['risk_type'].map(
       lambda x: risk_solution_map[x]['product']
   )
   ```

2. **ROI 예측**
   ```python
   # 개입 효과 시뮬레이션
   def calculate_roi(risk_type, avg_monthly_sales):
       solution = risk_solution_map[risk_type]

       # 비용 (평균)
       cost = {
           '마케팅 지원 대출': 500_000,
           '고객 리텐션 컨설팅 + CRM 시스템': 300_000,
           '오프라인 강화 프로그램': 1_000_000,
           '경쟁력 강화 패키지': 700_000,
           '경영 안정화 특별 패키지': 2_000_000
       }[solution['product']]

       # 예상 매출 증가 (6개월)
       expected_increase = {
           '매출 급락형': avg_monthly_sales * 0.15 * 6,
           '고객 이탈형': avg_monthly_sales * 0.10 * 6,
           '배달 의존형': avg_monthly_sales * 0.12 * 6,
           '경쟁 열위형': avg_monthly_sales * 0.08 * 6,
           '종합 위기형': avg_monthly_sales * 0.20 * 6  # 폐업 회피
       }[risk_type]

       roi = (expected_increase - cost) / cost * 100
       return roi

   df_recommendation['estimated_roi'] = df_recommendation.apply(
       lambda row: calculate_roi(row['risk_type'], row['avg_sales']),
       axis=1
   )
   ```

3. **조기 경보 성공 사례**
   ```python
   # 실제 폐업 가맹점 중 예측 성공 사례
   success_cases = df_test[
       (df_test['is_closed'] == 1) &
       (df_test['y_pred'] == 1)
   ].copy()

   # Lead time 분석
   success_cases['lead_time_days'] = (
       success_cases['MCT_ME_D'] -
       pd.to_datetime(success_cases['TA_YM'], format='%Y%m')
   ).dt.days

   print(f"조기 경보 성공률: {len(success_cases) / df_test['is_closed'].sum():.2%}")
   print(f"평균 리드 타임: {success_cases['lead_time_days'].mean():.0f}일")

   # 사례 정리
   for idx, case in success_cases.head(5).iterrows():
       print(f"\n[사례 {idx}]")
       print(f"업종: {case['HPSN_MCT_ZCD_NM']}")
       print(f"위험 유형: {case['risk_type']}")
       print(f"주요 위험 신호: {case['top_risk_signals']}")
       print(f"리드 타임: {case['lead_time_days']}일")
   ```

4. **최종 보고서 작성**
   ```markdown
   # 가맹점 위기 조기 경보 시스템 최종 보고서

   ## 1. Executive Summary
   - 모델 성능: PR-AUC 0.XX, Recall@10% 0.XX
   - 조기 경보 성공률: XX%
   - 평균 리드 타임: XX일

   ## 2. 주요 위기 신호 Top 10
   1. 재방문율 3개월 연속 하락
   2. 업종 내 매출 순위 20% 하락
   ...

   ## 3. 업종별 위험 패턴
   - 치킨: 배달 의존도 증가 + 객단가 하락
   - 카페: 신규 고객 유입 감소 + 경쟁 심화
   ...

   ## 4. 맞춤형 금융상품 제안
   ...

   ## 5. 기대효과
   - 폐업 예방: 연간 XX개 가맹점
   - 경제적 효과: XX억 원
   ```

5. **발표 자료 (PPT)**
   - 문제 정의 및 데이터 분석
   - 모델 아키텍처 및 성능
   - SHAP 분석 결과 (인사이트)
   - 비즈니스 제안
   - 기대효과 및 향후 계획

#### 산출물
- `notebooks/10_business_proposal.ipynb`
- `reports/final_report.pdf`
- `reports/presentation.pptx`
- `results/recommendation_by_merchant.csv`
- `results/roi_simulation.csv`
- `results/success_cases.csv`

---

## 4. 기술 스택

### 4.1 Core Libraries

```python
# 데이터 처리
import pandas as pd
import numpy as np
from datetime import datetime

# 모델링
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier  # 옵션

# 불균형 처리
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

# 하이퍼파라미터 튜닝
import optuna
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 앙상블
from sklearn.ensemble import VotingClassifier, StackingClassifier

# 평가
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    fbeta_score,
    roc_auc_score
)

# 해석
import shap
from lime import lime_tabular
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 유틸리티
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pickle
```

### 4.2 개발 환경

```toml
# pyproject.toml
[project]
name = "shinhan-202510"
version = "1.0.0"
requires-python = ">=3.12"

dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "imbalanced-learn>=0.11.0",
    "optuna>=3.3.0",
    "shap>=0.43.0",
    "lime>=0.2.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.17.0",
    "scipy>=1.11.0",
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "ipywidgets>=8.1.0"
]
```

---

## 5. 구현 세부사항

### 5.1 불균형 데이터 처리 전략

#### 전략 비교

| 방법 | 장점 | 단점 | 추천 |
|------|------|------|------|
| **SMOTE** | 소수 클래스 합성 생성 | 과적합 위험 | ⭐ |
| **ADASYN** | 경계 영역 집중 생성 | 노이즈 민감 | △ |
| **Tomek Links** | 경계 정리 | 데이터 손실 | - |
| **SMOTETomek** | 하이브리드 | 복잡성 증가 | ⭐ |
| **Class Weight** | 간단, 빠름 | 성능 제한적 | ⭐ |

#### 구현 코드

```python
# 1. SMOTE (추천)
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy=0.3,  # 소수:다수 = 3:10
    k_neighbors=5,
    random_state=42
)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts()}")
print(f"After SMOTE: {y_train_sm.value_counts()}")

# 2. Class Weight (XGBoost/LightGBM)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_params = {
    'scale_pos_weight': scale_pos_weight,  # ~32
    ...
}

lgb_params = {
    'is_unbalance': True,
    # 또는 'class_weight': 'balanced'
    ...
}

# 3. SMOTETomek (고급)
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)
```

### 5.2 시계열 특성 생성 베스트 프랙티스

#### Data Leakage 방지

```python
# ❌ 잘못된 예 (미래 정보 누출)
df['sales_ma_3m'] = df['sales'].rolling(3).mean()  # 미래 데이터 포함

# ✅ 올바른 예 (과거 데이터만 사용)
df['sales_ma_3m'] = df.groupby('ENCODED_MCT')['sales'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# 또는
df['sales_lag_1'] = df.groupby('ENCODED_MCT')['sales'].shift(1)
df['sales_ma_3m'] = df['sales_lag_1'].rolling(3, min_periods=1).mean()
```

#### 효율적인 특성 생성

```python
def create_time_features(df, target_col, mct_col='ENCODED_MCT'):
    """시계열 특성 일괄 생성"""

    features = df.copy()

    # Lag features
    for lag in [1, 3, 6]:
        features[f'{target_col}_lag_{lag}'] = features.groupby(mct_col)[target_col].shift(lag)

    # Rolling statistics
    for window in [3, 6]:
        features[f'{target_col}_ma_{window}'] = features.groupby(mct_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        features[f'{target_col}_std_{window}'] = features.groupby(mct_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Change rates
    features[f'{target_col}_change_1m'] = features.groupby(mct_col)[target_col].pct_change(1)
    features[f'{target_col}_change_3m'] = features.groupby(mct_col)[target_col].pct_change(3)

    # Consecutive patterns
    features[f'{target_col}_consecutive_up'] = features.groupby(mct_col)[f'{target_col}_change_1m'].apply(
        lambda x: (x > 0).astype(int).groupby((x <= 0).cumsum()).cumsum()
    )
    features[f'{target_col}_consecutive_down'] = features.groupby(mct_col)[f'{target_col}_change_1m'].apply(
        lambda x: (x < 0).astype(int).groupby((x >= 0).cumsum()).cumsum()
    )

    return features

# 사용 예
df = create_time_features(df, 'RC_M1_SAA')
df = create_time_features(df, 'RC_M1_UE_CUS_CN')
```

### 5.3 모델 학습 및 검증

#### Early Stopping 활용

```python
# XGBoost
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric=['aucpr', 'logloss'],
    early_stopping_rounds=50,
    verbose=50
)

# 최적 iteration 확인
best_iteration = xgb_model.best_iteration
print(f"Best iteration: {best_iteration}")

# LightGBM
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(50)
    ]
)
```

#### 시계열 교차 검증

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=3)  # 3개월씩 검증

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=30,
              verbose=False)

    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)

    cv_scores.append(pr_auc)
    print(f"PR-AUC: {pr_auc:.4f}")

print(f"\nAverage PR-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

### 5.4 앙상블 전략

#### Weighted Voting

```python
# 검증 성능 기반 가중치 계산
xgb_score = pr_auc_xgb  # 예: 0.65
lgb_score = pr_auc_lgb  # 예: 0.63

total_score = xgb_score + lgb_score
xgb_weight = xgb_score / total_score  # 0.51
lgb_weight = lgb_score / total_score  # 0.49

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    voting='soft',
    weights=[xgb_weight, lgb_weight]
)

voting_clf.fit(X_train, y_train)
y_proba_ensemble = voting_clf.predict_proba(X_test)[:, 1]
```

#### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Meta-learner에 클래스 가중치 적용
meta_learner = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    final_estimator=meta_learner,
    cv=TimeSeriesSplit(n_splits=5),
    passthrough=False  # Base model 예측만 사용
)

stacking_clf.fit(X_train, y_train)
```

---

## 6. 디렉토리 구조

```
shinhan_202510/
│
├── data/                           # 원본 데이터
│   ├── big_data_set1_f.csv
│   ├── big_data_set2_f.csv
│   └── big_data_set3_f.csv
│
├── notebooks/                      # Jupyter 노트북
│   ├── 00_temp.ipynb
│   ├── 01_eda.ipynb               # EDA (완료)
│   ├── 02_preprocessing.ipynb     # 데이터 전처리
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_xgboost.ipynb
│   ├── 05_model_lightgbm.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_ensemble.ipynb
│   ├── 08_shap_analysis.ipynb
│   ├── 09_insights.ipynb
│   └── 10_business_proposal.ipynb
│
├── pipeline/                      # ML 파이프라인 모듈
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # 데이터 로드
│   │   ├── feature_encoder.py    # 구간 인코딩
│   │   └── missing_handler.py    # 결측값 처리
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── time_series_features.py
│   │   ├── customer_features.py
│   │   ├── composite_features.py
│   │   └── feature_selector.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── ensemble.py
│   │   └── model_utils.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Custom metrics
│   │   └── validators.py         # CV 전략
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── shap_plots.py
│       ├── performance_plots.py
│       └── business_plots.py
│
├── models/                        # 학습된 모델 저장
│   ├── xgb_baseline.pkl
│   ├── lgb_baseline.pkl
│   ├── xgb_tuned.pkl
│   ├── lgb_tuned.pkl
│   ├── ensemble_voting.pkl
│   ├── ensemble_stacking.pkl
│   └── best_model.pkl
│
├── results/                       # 분석 결과
│   ├── feature_importance.csv
│   ├── shap_values.npy
│   ├── risk_patterns_by_industry.csv
│   ├── risk_type_classification.csv
│   ├── recommendation_by_merchant.csv
│   ├── roi_simulation.csv
│   ├── success_cases.csv
│   └── visualizations/
│       ├── shap_summary.png
│       ├── pr_curve.png
│       └── confusion_matrix.png
│
├── reports/                       # 최종 보고서
│   ├── final_report.pdf
│   ├── presentation.pptx
│   └── executive_summary.md
│
├── docs/                          # 문서
│   ├── 00_bigcontest_2025.md
│   ├── 01_data_layout.md
│   ├── 02_approach.md
│   └── 03_plan_XGBoost_LightGBM.md  # 이 문서
│
├── configs/                       # 설정 파일
│   ├── config.yaml
│   └── model_params.yaml
│
├── tests/                         # 테스트 코드
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── CLAUDE.md
└── main.py
```

---

## 7. 예상 결과물

### 7.1 모델 성능 목표

| 지표 | 목표값 | 달성 전략 |
|------|--------|-----------|
| **PR-AUC** | > 0.55 | 앙상블 + 시계열 특성 |
| **Recall@10%** | > 0.70 | 불균형 처리 + 임계값 최적화 |
| **Precision@10%** | > 0.60 | Feature Engineering |
| **F2-score** | > 0.60 | Recall 중시 튜닝 |
| **평균 리드 타임** | > 60일 | 조기 타겟 변수 (3개월 전) |

### 7.2 핵심 인사이트 (예상)

#### 주요 위기 신호 Top 10
1. **재방문율 3개월 연속 하락** (SHAP: 0.25)
2. **업종 내 매출 순위 20% 하락** (SHAP: 0.18)
3. **신규 고객 유입 6개월 감소** (SHAP: 0.15)
4. **배달 의존도 급증 + 총 매출 감소** (SHAP: 0.12)
5. **객단가 지속 하락** (SHAP: 0.10)
6. **상권 내 해지율 증가** (SHAP: 0.09)
7. **매출 변동성 급증** (SHAP: 0.08)
8. **운영 초기 6개월 + 매출 부진** (SHAP: 0.07)
9. **고객 다양성 감소** (SHAP: 0.06)
10. **유동인구 고객 비중 급감** (SHAP: 0.05)

#### 업종별 위험 패턴
- **치킨**: 배달 의존도 ↑ + 객단가 ↓
- **카페**: 신규 고객 유입 ↓ + 경쟁 심화
- **한식**: 재방문율 ↓ + 고객 고령화
- **일식**: 고가 메뉴 판매 ↓ + 변동성 ↑

### 7.3 비즈니스 제안

#### 위험 유형별 금융상품

| 위험 유형 | 추천 상품 | 기대효과 | 예상 ROI |
|-----------|----------|----------|----------|
| 매출 급락형 | 마케팅 지원 대출 | 매출 15% 회복 | 180% |
| 고객 이탈형 | CRM 시스템 지원 | 재방문율 20% ↑ | 150% |
| 배달 의존형 | 오프라인 강화 프로그램 | 오프라인 매출 30% ↑ | 120% |
| 경쟁 열위형 | 경쟁력 강화 패키지 | 순위 20% 상승 | 140% |
| 종합 위기형 | 경영 안정화 패키지 | 폐업 위험 50% ↓ | 250% |

#### 경제적 효과 (예상)
- **폐업 예방**: 연간 50~80개 가맹점
- **매출 회복**: 가맹점당 평균 500만원/월 × 6개월 = 3,000만원
- **총 경제적 효과**: 15억 ~ 24억 원/년
- **금융상품 대출액**: 가맹점당 평균 100만원 (총 5천만~8천만원)
- **예상 수익**: 대출 이자 + 컨설팅 수수료 = 1억 ~ 1.5억 원/년

### 7.4 최종 제출물

#### 1. 분석 보고서 (PDF)
- Executive Summary (2페이지)
- 데이터 분석 (10페이지)
- 모델링 결과 (10페이지)
- 인사이트 및 제안 (8페이지)

#### 2. 발표 자료 (PPT)
- 문제 정의 (3 slides)
- EDA 및 인사이트 (5 slides)
- 모델 아키텍처 (4 slides)
- SHAP 분석 결과 (5 slides)
- 비즈니스 제안 (5 slides)
- Q&A (2 slides)

#### 3. 코드 및 모델
- GitHub Repository
- 재현 가능한 Jupyter Notebooks
- 학습된 모델 파일 (.pkl)
- README with 실행 가이드

#### 4. 부록
- Feature 목록 및 정의
- 하이퍼파라미터 튜닝 로그
- 추가 시각화 자료

---

## 8. 위험 요소 및 대응 방안

### 8.1 잠재적 위험

| 위험 | 영향 | 확률 | 대응 방안 |
|------|------|------|-----------|
| **데이터 불균형 심화** | 성능 저하 | 높음 | SMOTE + Class Weight 병행 |
| **과적합** | 일반화 실패 | 중간 | Early Stopping + CV |
| **Feature 누출** | 비현실적 성능 | 중간 | Lag features만 사용 검증 |
| **해석 복잡도** | 인사이트 부족 | 낮음 | SHAP + 도메인 전문가 협업 |
| **계산 시간 부족** | 튜닝 제한 | 중간 | Optuna 효율적 탐색 |

### 8.2 대응 전략

1. **성능 모니터링**
   - Train/Valid/Test 성능 지속 추적
   - Overfitting 조기 발견

2. **백업 플랜**
   - XGBoost 단독 모델 확보
   - 간단한 Voting 앙상블 대안

3. **시간 관리**
   - 주차별 마일스톤 엄수
   - 우선순위 기반 작업

---

## 9. 성공 기준

### 9.1 기술적 성공

- [ ] PR-AUC > 0.55
- [ ] Recall@10% > 0.70
- [ ] 평균 리드 타임 > 60일
- [ ] SHAP 분석 완료
- [ ] 재현 가능한 코드

### 9.2 비즈니스 성공

- [ ] 위기 신호 Top 10 도출
- [ ] 업종별 패턴 5개 이상
- [ ] 금융상품 매칭 체계 구축
- [ ] ROI 시뮬레이션 완료
- [ ] 실행 가능한 제안

### 9.3 평가 기준 (100점)

- **모델링 적합성/완성도** (25점): XGBoost/LightGBM 앙상블, 불균형 처리
- **데이터 활용/분석력** (25점): 시계열 특성, 도메인 지식 활용
- **인사이트/실효성** (20점): SHAP 분석, 위기 신호 도출
- **금융상품 제안** (20점): 위험 유형별 맞춤 솔루션
- **완성도** (10점): 보고서 품질, 발표 완성도

---

## 10. 참고 자료

### 10.1 관련 논문
- Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
- Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions (SHAP)"

### 10.2 유용한 링크
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Imbalanced-learn Guide](https://imbalanced-learn.org/)
- [Optuna Tutorials](https://optuna.readthedocs.io/)

### 10.3 Kaggle 참고 솔루션
- [Imbalanced Classification - Credit Card Fraud](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
- [Time Series with XGBoost](https://www.kaggle.com/code/robikscube/time-series-forecasting-with-xgboost)
- [SHAP Analysis Tutorial](https://www.kaggle.com/code/dansbecker/shap-values)

---

**작성일**: 2025-10-05
**버전**: 1.0
**작성자**: AI Analysis Team
**프로젝트**: 2025 빅콘테스트 - 가맹점 위기 조기 경보 시스템
