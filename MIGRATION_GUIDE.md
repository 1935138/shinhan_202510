# Data Leakage 수정 마이그레이션 가이드

## 변경 사항 요약

**중요한 변경**: `is_closed` 변수가 data leakage를 유발하므로 제거되었습니다.

### 변경된 타겟 변수

| 변경 전 | 변경 후 | 설명 |
|---------|---------|------|
| `is_closed` | ❌ **제거됨** | 미래 정보 포함 (100% data leakage) |
| `will_close_1m` | ✅ 유지 | 1개월 내 폐업 예정 |
| `will_close_3m` | ✅ 유지 | 3개월 내 폐업 예정 (주 타겟) |
| `months_until_close` | ✅ 유지 | 참고용 (feature로 사용 금지) |
| (신규) | ✅ `is_valid_for_training` | 학습 데이터 필터용 |

## 마이그레이션 절차

### 1단계: 전처리 데이터 재생성 (필수)

기존 전처리 데이터는 `is_closed` 변수를 포함하고 있으므로 **반드시 재생성**해야 합니다.

```bash
# Jupyter notebook 실행
jupyter notebook

# 또는 VS Code에서
# notebooks/02_preprocessing.ipynb 열기
```

**`02_preprocessing.ipynb`를 처음부터 다시 실행하세요.**

새로운 출력 예시:
```
============================================================
CREATING TARGET VARIABLES (NO DATA LEAKAGE)
============================================================

Total records: 86,590
Valid for training: 84,256 (97.30%)
Invalid (already closed): 2,334 (2.70%)

Target distribution (valid data only):
will_close_1m = 1: 31 (0.04%)
will_close_3m = 1: 96 (0.11%)
```

### 2단계: Feature Engineering 데이터 재생성 (필수)

전처리 데이터가 변경되었으므로 feature engineering도 다시 실행해야 합니다.

```bash
# notebooks/03_feature_engineering.ipynb 실행
```

**변경 사항:**
- `is_closed` 변수가 입력 데이터에서 제거됨
- Feature 생성 로직은 동일

### 3단계: 모델 학습 코드 수정

기존 모델 학습 코드를 다음과 같이 수정하세요:

#### 변경 전:
```python
# ❌ 잘못된 방법
target_col = 'is_closed'

exclude_cols = [
    'ENCODED_MCT', 'TA_YM',
    'is_closed', 'will_close_1m', 'will_close_3m',
    'MCT_ME_D', 'months_until_close'
]

X = df.drop(columns=exclude_cols)
y = df[target_col]
```

#### 변경 후:
```python
# ✅ 올바른 방법
target_col = 'will_close_3m'  # 주 타겟

exclude_cols = [
    'ENCODED_MCT', 'TA_YM',
    'will_close_1m', 'will_close_3m',  # is_closed 제거됨
    'MCT_ME_D', 'months_until_close',
    'is_valid_for_training'  # 신규
]

# ✅ 필수: Valid 데이터만 사용
df_valid = df[df['is_valid_for_training'] == 1].copy()

X = df_valid.drop(columns=exclude_cols)
y = df_valid[target_col]
```

### 4단계: 검증 스크립트 실행

전처리 데이터가 올바르게 생성되었는지 확인합니다:

```bash
cd /home/dlsvud/project/shinhan_202510
python scripts/verify_no_leakage.py
```

**예상 출력:**
```
🎉 모든 검증을 통과했습니다!
   데이터가 data leakage 없이 올바르게 생성되었습니다.

⚠️  주의: 모델 학습 시 반드시 is_valid_for_training=1 데이터만 사용하세요!
```

## 영향받는 파일

### 수정 완료된 파일:
- ✅ `pipeline/preprocessing/feature_encoder.py`
- ✅ `docs/11_data_leakage_prevention.md` (신규)
- ✅ `CLAUDE.md`
- ✅ `scripts/verify_no_leakage.py` (신규)

### 재실행 필요한 노트북:
- 🔄 `notebooks/02_preprocessing.ipynb` - **필수 재실행**
- 🔄 `notebooks/03_feature_engineering.ipynb` - **필수 재실행**
- 🔄 `notebooks/04_model_training.ipynb` - 코드 수정 후 재실행
- 🔄 `notebooks/04-1_ensemble_model_training.ipynb` - 코드 수정 후 재실행
- 🔄 `notebooks/05_risk_prediction.ipynb` - 코드 수정 후 재실행

## FAQ

### Q1: 왜 `is_closed`가 data leakage인가요?

**A:** `is_closed`는 "이 가맹점은 미래에 폐업할 것이다"라는 정보를 모든 과거 데이터에 표시합니다.

예시:
```
가맹점 A가 202406에 폐업:
  202301: is_closed=1  ← 미래 정보! (5개월 후 폐업할 것을 이미 알고 있음)
  202302: is_closed=1  ← 미래 정보!
  ...
  202406: is_closed=1  ← 폐업 시점
```

모델이 `is_closed=1`만 보고 100% 정확하게 예측할 수 있지만, 실제 운영에서는 전혀 작동하지 않습니다.

### Q2: `will_close_3m`은 왜 괜찮나요?

**A:** `will_close_3m`은 **미래만 예측**하기 때문입니다.

```
가맹점 A가 202406에 폐업:
  202301: will_close_3m=0  ← 5개월 후 폐업 (3개월 초과)
  202302: will_close_3m=0  ← 4개월 후 폐업 (3개월 초과)
  202303: will_close_3m=1  ← 3개월 후 폐업 ✓
  202304: will_close_3m=1  ← 2개월 후 폐업 ✓
  202305: will_close_3m=1  ← 1개월 후 폐업 ✓
  202406: (제외)           ← 폐업 당월 데이터는 학습 제외
```

### Q3: `is_valid_for_training`은 무엇인가요?

**A:** 학습에 사용 가능한 데이터인지 표시하는 플래그입니다.

- `is_valid_for_training=1`: 학습 사용 가능
  - 영업 중인 가맹점
  - 폐업 예정이지만 아직 폐업 전
- `is_valid_for_training=0`: 학습 제외 (data leakage 위험)
  - 이미 폐업함 (폐업 당월 및 이후)
  - `months_until_close <= 0`

### Q4: 기존 모델 성능과 비교하면?

**A:** 기존 모델이 `is_closed`를 타겟으로 사용했다면:
- 기존: 매우 높은 정확도 (90-99%) - **가짜 성능** (data leakage)
- 새로운: 낮은 정확도 (60-80%) - **실제 성능** (no data leakage)

새로운 모델의 성능이 낮아 보일 수 있지만, 이것이 **실제 운영에서 작동하는 정확도**입니다.

### Q5: 이전 결과물은 어떻게 하나요?

**A:** 이전 결과물 (모델, 예측 결과 등)은 모두 **무효**입니다.
- `data/processed/preprocessed_data.csv` - 재생성 필요
- `data/processed/featured_data.csv` - 재생성 필요
- `models/*.pkl` - 재학습 필요
- `data/predictions/*.csv` - 재예측 필요

## 체크리스트

마이그레이션 완료 전 다음을 확인하세요:

- [ ] `02_preprocessing.ipynb` 재실행 완료
- [ ] `03_feature_engineering.ipynb` 재실행 완료
- [ ] `scripts/verify_no_leakage.py` 실행 및 통과
- [ ] 모델 학습 코드에서 `is_closed` 제거 및 `will_close_3m` 사용
- [ ] `is_valid_for_training=1` 필터링 추가
- [ ] 새로운 데이터로 모델 재학습
- [ ] Feature importance에 `months_until_close` 또는 `MCT_ME_D`가 상위에 없는지 확인

## 추가 참고 자료

- `docs/11_data_leakage_prevention.md` - 상세 가이드
- `CLAUDE.md` - 업데이트된 프로젝트 가이드
- `pipeline/preprocessing/feature_encoder.py` - 새로운 타겟 변수 생성 코드

## 문의

마이그레이션 중 문제가 발생하면:
1. `scripts/verify_no_leakage.py` 실행하여 문제 확인
2. `docs/11_data_leakage_prevention.md` 참조
3. 오류 메시지와 함께 이슈 제기
