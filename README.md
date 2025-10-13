# 가맹점 위기 조기 경보 시스템

**2025 빅콘테스트 AI/데이터 분석 프로젝트**

영세/중소 요식 가맹점의 경영 위기(매출 급락, 폐업 위험)를 1~3개월 전에 예측하는 AI 조기 경보 시스템입니다.

## 프로젝트 개요

- **대상**: 서울 성동구 요식업 가맹점 (4,185개)
- **데이터**: 24개월 거래 및 고객 데이터
- **모델**: XGBoost with All Interval Features (100+)
- **목표**: 폐업 위험 조기 감지 및 맞춤형 솔루션 제안

## 시스템 구성

```
┌─────────────────────┐         ┌──────────────────────┐
│   React Frontend    │────────▶│    Flask Backend     │
│   (Port 3000)       │◀────────│    (Port 5000)       │
└─────────────────────┘         └──────────────────────┘
                                          │
                                          ▼
                                 ┌──────────────────────┐
                                 │  XGBoost Model       │
                                 │  + Interval Features │
                                 └──────────────────────┘
```

## Quick Start

### 1. Backend 실행

```bash
# Python 환경 활성화
source .venv/bin/activate

# Flask API 서버 실행
python backend/app.py
```

서버가 `http://localhost:5000`에서 시작됩니다.

**초기 실행 시**: 모델이 없으면 자동으로 학습합니다 (~5분 소요)

### 2. Frontend 실행

**새 터미널에서**:

```bash
cd frontend

# Dependencies 설치 (처음만)
npm install

# React 앱 실행
npm start
```

브라우저가 자동으로 `http://localhost:3000`을 엽니다.

### 3. API 연동 활성화

`frontend/src/index.js` 수정:

```javascript
// API 버전 사용
import App from './AppWithAPI';

// 또는
import App from './App';  // 샘플 데이터 버전
```

## Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | 서버 상태 확인 |
| `/api/merchants` | GET | 가맹점 목록 (테스트용) |
| `/api/merchant/<id>/risk` | GET | 가맹점 위험도 예측 |
| `/api/merchant/<id>/signals` | GET | 위험 신호 감지 |
| `/api/merchant/<id>/sales` | GET | 매출 이력 (12개월) |
| `/api/regions/overview` | GET | 전체 가맹점 현황 |

### API 테스트

```bash
# Health check
curl http://localhost:5000/api/health

# 가맹점 목록
curl http://localhost:5000/api/merchants

# 특정 가맹점 위험도
curl "http://localhost:5000/api/merchant/MCT001/risk"
```

## Frontend 화면

1. **Home Screen**: 가맹점 위험도 점수 및 주요 위험 신호
2. **Detail Report**: 매출 추이, 고객 분석, 업종 비교
3. **Solution Screen**: 맞춤형 개선 솔루션 및 금융상품
4. **Web Dashboard**: 종합 대시보드 및 SHAP 분석
5. **Admin Console**: 전체 가맹점 현황 및 위험 가맹점 목록

## 핵심 기능: Interval Pattern Features

가맹점 성과 구간(1=최상위 ~ 6=최하위)의 변화 패턴을 분석하여 100개 이상의 피처 생성:

- **구간 하락 패턴** (36개): 연속 하락, 하락 속도, 기간별 하락 횟수
- **역대 최악 지표** (20개): 최악 구간, 최고 성과 대비 거리
- **회복 지표** (24개): 회복 패턴, 변동성, 방향 전환
- **교차 지표 비교** (20+개): 매출 vs 고객 괴리도, 동반 하락

### 모델 성능

**All Interval Features 모델** (최종 선택):
- Test ROC-AUC: 0.698
- Detection Rate: 4/22 (18.2%) - 최다 탐지
- 목표: Recall 최대화 (조기 경보 시스템)

자세한 내용: `docs/06_interval_feature_experiments_summary.md`

## 개발 환경

### Python (Backend)
```bash
# Python 3.12 + uv package manager
uv sync
source .venv/bin/activate
```

### Node.js (Frontend)
```bash
# Node.js 16+
cd frontend
npm install
```

## 프로젝트 구조

```
shinhan_202510/
├── backend/
│   └── app.py                  # Flask API 서버
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # 샘플 데이터 버전
│   │   ├── AppWithAPI.jsx     # API 연동 버전
│   │   ├── services/api.js    # API 호출 로직
│   │   └── screens/           # 화면 컴포넌트
│   └── package.json
├── notebooks/
│   ├── 03-1_interval_pattern_features.ipynb
│   └── 04-1_model_training_with_interval_feature.ipynb  # 최종 모델
├── pipeline/
│   └── features/
│       └── interval_patterns.py  # 100+ interval features
├── data/
│   ├── raw/                   # 원본 데이터
│   └── processed/             # 전처리된 데이터
├── models/                    # 학습된 모델 파일
└── docs/                      # 프로젝트 문서
    ├── 05_interval_pattern_features.md
    ├── 06_interval_feature_experiments_summary.md
    └── 07_webapp_integration_guide.md
```

## 트러블슈팅

### Backend가 시작되지 않음
```bash
source .venv/bin/activate
uv sync
python backend/app.py
```

### Frontend가 API를 호출하지 않음
`frontend/src/index.js`에서 `AppWithAPI` 사용 확인

### 데이터 파일 없음
```bash
jupyter notebook notebooks/03-1_interval_pattern_features.ipynb
```

## 상세 문서

- **웹앱 통합 가이드**: `docs/07_webapp_integration_guide.md`
- **Interval Features 전략**: `docs/05_interval_pattern_features.md`
- **실험 결과 요약**: `docs/06_interval_feature_experiments_summary.md`
- **프로젝트 가이드**: `CLAUDE.md`

## 라이선스

2025 빅콘테스트 출품작

---

**최종 업데이트**: 2025-10-13  
**개발 환경**: Python 3.12 + React 19 + XGBoost
