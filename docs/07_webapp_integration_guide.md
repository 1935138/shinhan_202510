# 웹앱 통합 가이드: All Interval Features Model

## 개요

이 문서는 All Interval Features XGBoost 모델을 웹앱(React Frontend + Flask Backend)에 통합한 내용을 설명합니다.

**시스템 구성**:
- **Backend**: Flask API (Python) - 모델 서빙 및 예측
- **Frontend**: React App (JavaScript) - 사용자 인터페이스
- **Model**: XGBoost with 100+ Interval Pattern Features

---

## 시스템 아키텍처

```
┌─────────────────────┐         ┌──────────────────────┐
│   React Frontend    │────────▶│    Flask Backend     │
│   (Port 3000)       │◀────────│    (Port 5000)       │
└─────────────────────┘         └──────────────────────┘
         │                                  │
         │                                  │
         ▼                                  ▼
    User Browser                 ┌──────────────────────┐
                                 │  XGBoost Model       │
                                 │  + Feature Data      │
                                 └──────────────────────┘
```

---

## Backend API (Flask)

### 파일 구조

```
backend/
└── app.py                  # Flask API 서버
```

### API Endpoints

#### 1. Health Check
```
GET /api/health
```
**응답**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_loaded": true,
  "timestamp": "2025-10-13T..."
}
```

#### 2. List Merchants (테스트용)
```
GET /api/merchants
```
**응답**:
```json
{
  "total": 4185,
  "merchants": ["MCT001", "MCT002", ...],
  "timestamp": "..."
}
```

#### 3. Get Merchant Risk
```
GET /api/merchant/<merchant_id>/risk
GET /api/merchant/<merchant_id>/risk?date=202412
```
**응답**:
```json
{
  "merchant_id": "MCT001",
  "category": "한식",
  "date": 202412,
  "risk_score": 72,
  "risk_probability": 0.7234,
  "status": "warning",
  "days_until_crisis": 45,
  "timestamp": "..."
}
```

**Status 분류**:
- `danger`: risk_score >= 80 (30일 이내 위기)
- `warning`: risk_score >= 60 (45일 이내 위기)
- `caution`: risk_score >= 40 (60일 이내 위기)
- `safe`: risk_score < 40

#### 4. Get Risk Signals
```
GET /api/merchant/<merchant_id>/signals
```
**응답**:
```json
{
  "merchant_id": "MCT001",
  "signals": [
    {
      "icon": "trending-down",
      "title": "3개월 연속 성과 하락",
      "description": "RC_M1_SAA 지표가 지속적으로 악화",
      "color": "red"
    },
    {
      "icon": "alert-triangle",
      "title": "역대 최악 성과 기록 중",
      "description": "2개 지표가 역대 최저",
      "color": "red"
    }
  ],
  "timestamp": "..."
}
```

**Risk Signal 감지 로직**:
1. **연속 하락**: `consecutive_decline >= 3`
2. **역대 최악**: `at_worst_now >= 2` 지표
3. **전성기 대비 하락**: `distance_from_best >= 3`
4. **복합 하락**: `aligned_decline >= 1`
5. **높은 변동성**: `volatility > 1.5` (2개 이상)

#### 5. Get Sales History
```
GET /api/merchant/<merchant_id>/sales
```
**응답**:
```json
{
  "merchant_id": "MCT001",
  "sales_data": [
    {"month": "01월", "sales": 4000},
    {"month": "02월", "sales": 3800},
    ...
  ],
  "timestamp": "..."
}
```

**참고**: 실제 매출 데이터 대신 `RC_M1_SAA` interval을 inverse하여 상대적 매출 점수로 변환합니다.

#### 6. Get Region Overview
```
GET /api/regions/overview
```
**응답**:
```json
{
  "region_name": "성동구",
  "total_stores": 4185,
  "status_breakdown": [
    {"status": "안전", "count": 2500, "percentage": 60.0, "color": "safe"},
    {"status": "양호", "count": 1000, "percentage": 24.0, "color": "good"},
    {"status": "주의", "count": 500, "percentage": 12.0, "color": "caution"},
    {"status": "경고", "count": 150, "percentage": 3.5, "color": "warning"},
    {"status": "위험", "count": 35, "percentage": 0.8, "color": "danger"}
  ],
  "risk_merchants": [
    {
      "merchant_id": "MCT001",
      "category": "한식",
      "risk_score": 92,
      "risk_type": "매출급락형",
      "is_urgent": true
    },
    ...
  ],
  "timestamp": "..."
}
```

---

## Frontend (React)

### 파일 구조

```
frontend/src/
├── App.jsx                      # 기존 샘플 데이터 버전
├── AppWithAPI.jsx               # API 연동 버전 (신규)
├── services/
│   └── api.js                   # API 호출 로직
└── screens/
    ├── HomeScreen.jsx           # 홈 화면 (위험도 점수)
    ├── DetailReportScreen.jsx   # 상세 리포트
    ├── SolutionScreen.jsx       # 솔루션 제안
    ├── WebDashboardScreen.jsx   # 웹 대시보드
    └── AdminConsoleScreen.jsx   # 관리자 콘솔
```

### API Service (`services/api.js`)

**주요 함수**:
```javascript
// API 호출
getMerchantRisk(merchantId, date)
getMerchantSignals(merchantId)
getMerchantSales(merchantId)
getRegionOverview()
listMerchants()

// 데이터 변환 (API → Screen)
transformRiskDataForHome(riskData, signalsData)
transformSalesDataForDetail(salesData, riskData)
transformDataForDashboard(riskData, salesData, signalsData)
transformDataForAdmin(overviewData)
```

### AppWithAPI.jsx

**기능**:
- 초기 로딩 시 샘플 가맹점 데이터 로드
- 화면 전환 시 필요한 데이터 자동 로드
- 로딩/에러 상태 처리
- 관리자 화면에서 가맹점 클릭 시 해당 가맹점 데이터 로드

**주요 State**:
```javascript
const [currentScreen, setCurrentScreen] = useState('home');
const [currentMerchant, setCurrentMerchant] = useState(null);
const [homeData, setHomeData] = useState(null);
const [detailData, setDetailData] = useState(null);
const [dashboardData, setDashboardData] = useState(null);
const [adminData, setAdminData] = useState(null);
```

---

## 실행 방법

### 1. Backend 실행

```bash
# 프로젝트 루트에서
cd /home/dlsvud/project/shinhan_202510

# Python 환경 활성화
source .venv/bin/activate

# Flask 서버 실행
python backend/app.py
```

**실행 시 로그**:
```
✅ Loaded data: (87826, 200)
✅ Feature columns: 150
⚠️  Model not found. Training new model...
================================
TRAINING ALL INTERVAL FEATURES MODEL
================================
Train: 50000 rows, Positive: 120
Valid: 20000 rows, Positive: 40
Scale positive weight: 416.7
Training model...
✅ Model saved to: models/xgboost_all_interval_model.pkl
================================
🚀 Starting Flask API Server
================================
 * Running on http://0.0.0.0:5000
```

### 2. Frontend 실행

**새 터미널에서**:
```bash
cd /home/dlsvud/project/shinhan_202510/frontend

# Dependencies 설치 (처음만)
npm install

# React 앱 실행
npm start
```

브라우저가 자동으로 열리고 `http://localhost:3000`에서 앱이 실행됩니다.

### 3. API 버전 사용

**기본 `App.jsx` 대신 `AppWithAPI.jsx` 사용**:

**방법 1**: `index.js` 수정
```javascript
// frontend/src/index.js
import AppWithAPI from './AppWithAPI';  // 변경

ReactDOM.render(<AppWithAPI />, document.getElementById('root'));
```

**방법 2**: `App.jsx` 교체
```bash
cd frontend/src
mv App.jsx App.sample.jsx
mv AppWithAPI.jsx App.jsx
```

---

## 통합 테스트

### Test 1: Health Check
```bash
curl http://localhost:5000/api/health
```

**Expected**:
```json
{"status": "healthy", "model_loaded": true, ...}
```

### Test 2: Get Merchant List
```bash
curl http://localhost:5000/api/merchants
```

**Expected**:
```json
{"total": 4185, "merchants": [...]}
```

### Test 3: Get Merchant Risk
```bash
# 첫 번째 가맹점 ID 사용
curl http://localhost:5000/api/merchant/<MERCHANT_ID>/risk
```

**Expected**:
```json
{"merchant_id": "...", "risk_score": 72, "status": "warning", ...}
```

### Test 4: Frontend 동작 확인

1. **Home Screen**: 위험도 점수 및 신호 표시
2. **Detail Report**: 매출 차트 및 비교 지표
3. **Solution**: 개선 솔루션 및 금융상품
4. **Dashboard**: 웹 대시보드 (SHAP 포함)
5. **Admin Console**: 전체 가맹점 현황

---

## 데이터 흐름

### 홈 화면 렌더링 예시

```
1. AppWithAPI.jsx 초기화
   └─▶ loadSampleMerchant()
        └─▶ listMerchants() [API]
             └─▶ 첫 번째 가맹점 선택

2. loadMerchantData(merchantId)
   ├─▶ getMerchantRisk(merchantId) [API]
   ├─▶ getMerchantSignals(merchantId) [API]
   └─▶ getMerchantSales(merchantId) [API]

3. 데이터 변환
   └─▶ transformRiskDataForHome(riskData, signalsData)
        └─▶ setHomeData({healthData, riskSignals})

4. HomeScreen 렌더링
   └─▶ healthData.score: 72
   └─▶ riskSignals: ["3개월 연속 하락", ...]
```

---

## 모델 학습 프로세스

### 자동 학습 (Model Not Found)

Backend 실행 시 모델 파일이 없으면 자동으로 학습:

```python
# backend/app.py의 train_model()
1. featured_data에서 valid training data 필터링
2. Train/Valid split (202301-202406 / 202407-202409)
3. XGBoost 학습
   - scale_pos_weight: 자동 계산
   - early_stopping_rounds: 50
   - eval_metric: aucpr
4. models/xgboost_all_interval_model.pkl 저장
```

### 수동 학습 (Notebook 사용)

더 정교한 학습이 필요한 경우:

```bash
cd notebooks
jupyter notebook 04-1_model_training_with_interval_feature.ipynb
```

학습 후 모델을 `models/` 디렉토리에 저장하면 Backend에서 자동 로드합니다.

---

## 환경 변수

### Backend (Optional)

```bash
# backend/.env
DATA_DIR=/path/to/data/processed
MODEL_DIR=/path/to/models
```

### Frontend

```bash
# frontend/.env
REACT_APP_API_URL=http://localhost:5000/api
```

---

## 트러블슈팅

### 1. Backend가 시작되지 않음

**증상**: `ModuleNotFoundError: No module named 'flask'`

**해결**:
```bash
source .venv/bin/activate
uv sync
```

### 2. 모델 학습이 너무 오래 걸림

**증상**: Training 단계에서 10분 이상 소요

**해결**:
```python
# backend/app.py의 XGBClassifier
n_estimators=200,  # 500 → 200으로 축소
```

### 3. CORS 에러

**증상**: `Access-Control-Allow-Origin` 에러

**해결**: Flask-CORS가 이미 설정되어 있으므로 Backend 재시작
```bash
# backend/app.py
CORS(app)  # 이미 설정됨
```

### 4. 데이터 파일 없음

**증상**: `FileNotFoundError: data/processed/featured_data_with_intervals.csv`

**해결**:
```bash
# Notebook 03-1 실행하여 데이터 생성
jupyter notebook notebooks/03-1_interval_pattern_features.ipynb
```

### 5. Frontend가 API를 호출하지 않음

**증상**: 계속 샘플 데이터만 표시

**해결**: `index.js`에서 `AppWithAPI` import 확인
```javascript
import App from './AppWithAPI';  // 또는 App.jsx를 AppWithAPI.jsx로 교체
```

---

## 성능 최적화

### Backend

1. **모델 캐싱**: 모델을 global 변수로 한 번만 로드
2. **데이터 필터링**: 최신 N개월 데이터만 메모리 유지
3. **병렬 처리**: 여러 가맹점 예측 시 batch prediction 사용

### Frontend

1. **데이터 캐싱**: 이미 로드한 가맹점 데이터 재사용
2. **Lazy Loading**: 화면 전환 시에만 필요한 데이터 로드
3. **Debouncing**: 검색/필터 시 과도한 API 호출 방지

---

## 다음 단계

### 1. SHAP 분석 추가
- Backend에 `/api/merchant/<id>/shap` 엔드포인트 추가
- 실제 SHAP values 계산 및 반환

### 2. 위험 유형 분류
- Interval pattern features를 분석하여 위험 유형 자동 분류
- "매출급락형", "고객이탈형", "배달의존형" 등

### 3. 실시간 알림
- WebSocket 또는 Server-Sent Events로 위험 신호 실시간 푸시
- 새로운 위험 가맹점 감지 시 알림

### 4. 데이터베이스 연동
- SQLite/PostgreSQL로 가맹점 메타데이터 관리
- 컨설팅 이력, 솔루션 적용 결과 저장

### 5. 인증/권한
- JWT 기반 인증 추가
- 가맹점 소유자 vs 관리자 권한 분리

---

## 참고 문서

- Backend API 코드: `backend/app.py`
- Frontend API Service: `frontend/src/services/api.js`
- Frontend App (API 버전): `frontend/src/AppWithAPI.jsx`
- Interval Features 문서: `docs/05_interval_pattern_features.md`
- 실험 결과 요약: `docs/06_interval_feature_experiments_summary.md`

---

**작성일**: 2025-10-13
**작성자**: Claude Code
**버전**: 1.0
