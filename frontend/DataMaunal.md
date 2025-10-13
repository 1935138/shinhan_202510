# 신한 가맹점 헬스케어 - 데이터 연동 가이드

## 📋 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [설치 가이드](#설치-가이드)
3. [프로젝트 구조](#프로젝트-구조)
4. [데이터 형식 (JSON 스키마)](#데이터-형식)
5. [데이터 연결 방법](#데이터-연결-방법)
6. [Python 백엔드 연동](#python-백엔드-연동)
7. [예제 코드](#예제-코드)

---

## 💻 시스템 요구사항

### 프론트엔드 (React)
- **Node.js**: 16.x 이상
- **npm**: 8.x 이상 (Node.js 설치 시 자동 포함)
- **운영체제**: macOS, Windows, Linux

### 백엔드 (Python)
- **Python**: 3.8 이상
- **pip**: 최신 버전
- **운영체제**: macOS, Windows, Linux

---

## 🛠️ 설치 가이드

### 🔥 Git에서 프로젝트 받은 경우 (가장 일반적)

#### 1️⃣ 프로젝트 클론

```bash
# Git 저장소에서 클론
git clone <repository-url>
cd shinhan-healthcare
```

#### 2️⃣ 프론트엔드 설치 및 실행

```bash
# 프로젝트 폴더에서
npm install              # ⭐ 필수! node_modules 설치
npm start                # 개발 서버 실행
```

**⚠️ 중요:**
- `node_modules` 폴더는 Git에 포함되지 않습니다 (.gitignore)
- **반드시 `npm install`을 먼저 실행**해야 합니다!
- 약 2-5분 정도 소요됩니다.

#### 3️⃣ 백엔드 설치 및 실행 (있는 경우)

```bash
cd backend

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 라이브러리 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload   # FastAPI
# 또는
python app.py               # Flask
```

#### 4️⃣ 환경 변수 설정 (.env 파일)

프로젝트 루트에 `.env` 파일 생성:

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENV=development
```

---


### 🆕 처음부터 새로 시작하는 경우

#### 1️⃣ 프론트엔드 설치

#### 1-1. Node.js 설치

**macOS:**
```bash
# Homebrew 사용 (추천)
brew install node

# 또는 공식 웹사이트에서 다운로드
# https://nodejs.org
```

**Windows:**
```bash
# 공식 웹사이트에서 다운로드
# https://nodejs.org
# 설치 후 터미널 재시작
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### 1-2. 설치 확인

```bash
node --version   # v16.0.0 이상
npm --version    # 8.0.0 이상
```

#### 1-3. React 프로젝트 생성

```bash
# 원하는 디렉토리로 이동
cd ~/Documents

# React 프로젝트 생성
npx create-react-app shinhan-healthcare

# 프로젝트 디렉토리로 이동
cd shinhan-healthcare
```

#### 1-4. 필수 라이브러리 설치

```bash
# 차트 라이브러리
npm install recharts

# 아이콘 라이브러리
npm install lucide-react

# Tailwind CSS (CSS 프레임워크)
npm install -D tailwindcss@3.4.1 postcss autoprefixer

# Tailwind 초기화
npx tailwindcss init -p
```

**또는 한 번에 설치:**
```bash
npm install recharts lucide-react
npm install -D tailwindcss@3.4.1 postcss autoprefixer
npx tailwindcss init -p
```

#### 1-5. package.json 예시

프로젝트 루트의 `package.json` 파일은 다음과 같아야 합니다:

```json
{
  "name": "shinhan-healthcare",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.10.3",
    "lucide-react": "^0.263.1"
  },
  "devDependencies": {
    "tailwindcss": "^3.4.1",
    "postcss": "^8.4.35",
    "autoprefixer": "^10.4.17"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }
}
```

#### 1-6. Tailwind CSS 설정

**tailwind.config.js** (프로젝트 루트):
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**postcss.config.js** (프로젝트 루트):
```javascript
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

**src/index.css** (맨 위에 추가):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

### 2️⃣ 백엔드 설치 (Python)

#### 2-1. Python 설치

**macOS:**
```bash
# Homebrew 사용
brew install python@3.11

# 또는 공식 웹사이트에서 다운로드
# https://www.python.org
```

**Windows:**
```bash
# 공식 웹사이트에서 다운로드
# https://www.python.org
# 설치 시 "Add Python to PATH" 체크!
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

#### 2-2. 설치 확인

```bash
python3 --version  # 3.8 이상
pip3 --version
```

#### 2-3. 가상환경 생성 (권장)

```bash
# 백엔드 디렉토리 생성
mkdir backend
cd backend

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### 2-4. 필수 라이브러리 설치

**FastAPI 사용 시:**
```bash
pip install fastapi uvicorn python-multipart pydantic
pip install pandas numpy
```

**Flask 사용 시:**
```bash
pip install flask flask-cors
pip install pandas numpy
```

**공통 라이브러리:**
```bash
pip install python-dotenv requests
```

#### 2-5. requirements.txt 생성

**FastAPI용 requirements.txt:**
```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pydantic==2.5.3
pandas==2.1.4
numpy==1.26.3
python-dotenv==1.0.0
```

**Flask용 requirements.txt:**
```txt
flask==3.0.0
flask-cors==4.0.0
pandas==2.1.4
numpy==1.26.3
python-dotenv==1.0.0
```

**requirements.txt로 설치:**
```bash
pip install -r requirements.txt
```

---

### 3️⃣ 프로젝트 구조 설정

```bash
shinhan-healthcare/
├── frontend/                    # React 프로젝트
│   ├── public/
│   │   ├── index.html
│   │   └── data/               # JSON 파일 (테스트용)
│   │       ├── home.json
│   │       ├── detailReport.json
│   │       └── solution.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.js
│   │   ├── index.css
│   │   ├── screens/
│   │   │   ├── HomeScreen.jsx
│   │   │   ├── DetailReportScreen.jsx
│   │   │   ├── SolutionScreen.jsx
│   │   │   ├── WebDashboardScreen.jsx
│   │   │   └── AdminConsoleScreen.jsx
│   │   └── services/
│   │       ├── api.js           # API 호출 함수
│   │       └── dataLoader.js    # JSON 로더
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── package.json
│
└── backend/                     # Python 백엔드
    ├── venv/                    # 가상환경 (git ignore)
    ├── main.py                  # FastAPI 메인
    ├── requirements.txt
    ├── .env                     # 환경 변수
    └── data/                    # CSV/JSON 데이터
        └── merchants.csv
```

---

### 4️⃣ 빠른 시작 (Quick Start)

#### 🎯 Git Clone 받은 경우 (추천)

**1. 저장소 클론 및 프론트엔드 실행:**
```bash
# 프로젝트 클론
git clone <repository-url>
cd shinhan-healthcare

# 의존성 설치 (⭐ 필수! 약 2-5분 소요)
npm install

# Tailwind CSS 설정 확인 (이미 있으면 skip)
# tailwind.config.js와 postcss.config.js 파일 확인

# 개발 서버 실행
npm start
```

브라우저가 자동으로 열리고 http://localhost:3000 에서 앱 실행!

**2. 백엔드 실행 (별도 터미널):**
```bash
cd backend

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload  # http://localhost:8000
```

---

#### 🔧 처음부터 프로젝트 생성하는 경우

**프론트엔드 실행:**

```bash
cd frontend
npm install                 # 의존성 설치 (최초 1회)
npm start                   # 개발 서버 실행 (http://localhost:3000)
```

**백엔드 실행 (FastAPI):**

```bash
cd backend
source venv/bin/activate    # 가상환경 활성화
pip install -r requirements.txt  # 의존성 설치 (최초 1회)
uvicorn main:app --reload   # 개발 서버 실행 (http://localhost:8000)
```

**백엔드 실행 (Flask):**

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
python app.py               # 개발 서버 실행
```

---

### 📝 체크리스트 (다른 개발자가 Git Clone 후)

프로젝트를 Git에서 받은 후 다음을 확인하세요:

**프론트엔드:**
- [ ] `npm install` 실행했는가?
- [ ] `node_modules/` 폴더가 생성되었는가?
- [ ] `tailwind.config.js` 파일이 있는가?
- [ ] `postcss.config.js` 파일이 있는가?
- [ ] `src/index.css`에 `@tailwind` 지시어가 있는가?
- [ ] `.env` 파일 생성했는가? (필요시)
- [ ] `npm start` 실행 시 에러 없이 실행되는가?

**백엔드:**
- [ ] `python3 -m venv venv` 실행했는가?
- [ ] 가상환경 활성화했는가?
- [ ] `pip install -r requirements.txt` 실행했는가?
- [ ] `.env` 파일 생성했는가? (필요시)
- [ ] 서버 실행 시 에러 없이 실행되는가?

---

### 5️⃣ 설치 문제 해결

#### Node.js/npm 문제

**권한 에러 (macOS/Linux):**
```bash
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules
```

**캐시 정리:**
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### Python/pip 문제

**pip 업그레이드:**
```bash
pip install --upgrade pip
```

**가상환경 재생성:**
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Tailwind CSS가 적용 안 될 때

**방법 1: CDN 사용 (빠른 해결)**

`public/index.html`의 `<head>` 안에:
```html
<script src="https://cdn.tailwindcss.com"></script>
```

`src/index.css`에서 `@tailwind` 지시어 제거

**방법 2: 완전 재설치**
```bash
npm uninstall tailwindcss postcss autoprefixer
npm install -D tailwindcss@3.4.1 postcss autoprefixer
npx tailwindcss init -p
```

---

## 🏗️ 프로젝트 구조

```
src/
├── App.jsx                    # 메인 라우팅 및 데이터 관리
├── screens/
│   ├── HomeScreen.jsx         # 홈 화면
│   ├── DetailReportScreen.jsx # 상세 리포트
│   ├── SolutionScreen.jsx     # 추천 솔루션
│   ├── WebDashboardScreen.jsx # 웹 대시보드
│   └── AdminConsoleScreen.jsx # 관리자 콘솔
└── services/
    └── api.js                 # API 호출 함수 (추가 필요)
```

---

## 📊 데이터 형식 (JSON 스키마)

### 1. HomeScreen 데이터

```json
{
  "healthData": {
    "score": 72,
    "status": "warning",        // "good" | "warning" | "danger"
    "daysUntilCrisis": 45
  },
  "riskSignals": [
    {
      "icon": "repeat",         // "repeat" | "trending" | "package"
      "title": "재방문 고객 20% 감소",
      "description": "지난 달 대비 재방문율 급격히 하락",
      "color": "red"            // "red" | "orange" | "amber"
    }
  ]
}
```

### 2. DetailReportScreen 데이터

```json
{
  "salesData": [
    { "month": "4월", "sales": 3200 },
    { "month": "5월", "sales": 3800 }
  ],
  "customerData": {
    "returning": 45,
    "new": 55,
    "ageDistribution": [
      { "age": "20대", "percentage": 15 },
      { "age": "30대", "percentage": 35 }
    ]
  },
  "comparisonData": {
    "industryAverage": 85,
    "ranking": 15,
    "totalStores": 50,
    "additionalMetrics": [
      { "label": "전월 대비", "value": "-5%" }
    ]
  }
}
```

### 3. SolutionScreen 데이터

```json
{
  "improvementSolutions": [
    {
      "priority": 1,
      "title": "고객 재방문 캠페인",
      "description": "기존 고객 대상 맞춤형 할인 쿠폰 제공",
      "expectedEffect": "재방문 15% ↑",
      "cost": "30만원",
      "duration": "1개월",
      "badge": "긴급"           // optional
    }
  ],
  "financialProducts": [
    {
      "name": "마케팅 지원 대출",
      "description": "가맹점 마케팅 활동을 위한 특별 금리 대출",
      "interestRate": "3.5%",
      "maxAmount": "최대 500만원",
      "badge": "추천",          // optional
      "additionalInfo": [
        { "label": "상환 기간", "value": "최대 12개월" }
      ],
      "benefits": [
        "마케팅 비용 사용 시 금리 우대",
        "3개월 거치 가능"
      ]
    }
  ]
}
```

### 4. WebDashboardScreen 데이터

```json
{
  "healthData": {
    "score": 72,
    "status": "warning",
    "daysUntilCrisis": 45
  },
  "salesData": [
    { "month": "4월", "sales": 3200 },
    // ... 12개월 데이터
  ],
  "customerData": {
    "returning": 45,
    "new": 55
  },
  "shapData": [
    { "label": "재방문율 감소", "value": 0.25 },
    { "label": "업종 순위 하락", "value": 0.18 }
  ]
}
```

### 5. AdminConsoleScreen 데이터

```json
{
  "regionName": "성동구",
  "totalStores": 4185,
  "statusBreakdown": [
    {
      "status": "안전",
      "count": 2500,
      "percentage": 60,
      "color": "safe"          // "safe" | "good" | "caution" | "warning" | "danger"
    }
  ],
  "riskMerchants": [
    {
      "name": "호호치킨",
      "category": "치킨",
      "riskScore": 92,
      "riskType": "매출급락형",  // "매출급락형" | "고객이탈형" | "경쟁열위형"
      "lastConsultDate": "2024-09-15", // or null
      "isUrgent": true
    }
  ]
}
```

---

## 🔌 데이터 연결 방법

### 방법 1: JSON 파일 사용 (개발/테스트)

#### 1-1. JSON 파일 생성

```
public/data/
├── home.json
├── detailReport.json
├── solution.json
├── webDashboard.json
└── adminConsole.json
```

#### 1-2. 데이터 로드 함수 생성

`src/services/dataLoader.js`:

```javascript
export const loadJSON = async (filename) => {
  try {
    const response = await fetch(`/data/${filename}`);
    if (!response.ok) throw new Error('데이터 로드 실패');
    return await response.json();
  } catch (error) {
    console.error('JSON 로드 에러:', error);
    return null;
  }
};
```

#### 1-3. App.jsx에서 사용

```javascript
import { useState, useEffect } from 'react';
import { loadJSON } from './services/dataLoader';

export default function App() {
  const [homeData, setHomeData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const data = await loadJSON('home.json');
      setHomeData(data);
      setLoading(false);
    };
    fetchData();
  }, []);

  if (loading) return <div>로딩 중...</div>;

  return <HomeScreen {...homeData} />;
}
```

---

### 방법 2: REST API 연동 (프로덕션)

#### 2-1. API 서비스 생성

`src/services/api.js`:

```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 공통 fetch 함수
const fetchAPI = async (endpoint, options = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    
    if (!response.ok) {
      throw new Error(`API 에러: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API 호출 실패:', error);
    throw error;
  }
};

// 각 화면별 API 함수
export const api = {
  // 홈 화면 데이터
  getHomeData: (merchantId) => 
    fetchAPI(`/api/merchants/${merchantId}/home`),
  
  // 상세 리포트 데이터
  getDetailReport: (merchantId) => 
    fetchAPI(`/api/merchants/${merchantId}/report`),
  
  // 추천 솔루션 데이터
  getSolutions: (merchantId) => 
    fetchAPI(`/api/merchants/${merchantId}/solutions`),
  
  // 웹 대시보드 데이터
  getDashboard: (merchantId) => 
    fetchAPI(`/api/merchants/${merchantId}/dashboard`),
  
  // 관리자 콘솔 데이터
  getAdminData: (regionName) => 
    fetchAPI(`/api/admin/regions/${regionName}`),
  
  // 가맹점 목록
  getMerchants: () => 
    fetchAPI('/api/merchants'),
};
```

#### 2-2. App.jsx에서 API 사용

```javascript
import { useState, useEffect } from 'react';
import { api } from './services/api';

export default function App() {
  const [homeData, setHomeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const merchantId = '12345'; // 실제로는 로그인 정보에서 가져옴

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const data = await api.getHomeData(merchantId);
        setHomeData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [merchantId]);

  if (loading) return <div>로딩 중...</div>;
  if (error) return <div>에러: {error}</div>;

  return <HomeScreen {...homeData} />;
}
```

---

## 🐍 Python 백엔드 연동

### FastAPI 예제

`backend/main.py`:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI()

# CORS 설정 (React 개발 서버와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델 정의
class HealthData(BaseModel):
    score: int
    status: str
    daysUntilCrisis: int

class RiskSignal(BaseModel):
    icon: str
    title: str
    description: str
    color: str

class HomeData(BaseModel):
    healthData: HealthData
    riskSignals: List[RiskSignal]

# API 엔드포인트
@app.get("/api/merchants/{merchant_id}/home", response_model=HomeData)
async def get_home_data(merchant_id: str):
    """
    홈 화면 데이터 반환
    """
    # 실제로는 DB에서 데이터를 가져옴
    # 여기서는 예제 데이터 반환
    
    return {
        "healthData": {
            "score": 72,
            "status": "warning",
            "daysUntilCrisis": 45
        },
        "riskSignals": [
            {
                "icon": "repeat",
                "title": "재방문 고객 20% 감소",
                "description": "지난 달 대비 재방문율 급격히 하락",
                "color": "red"
            }
        ]
    }

@app.get("/api/merchants/{merchant_id}/report")
async def get_detail_report(merchant_id: str):
    """
    상세 리포트 데이터 반환
    """
    # CSV/JSON 파일 읽기 또는 DB 쿼리
    return {
        "salesData": [...],
        "customerData": {...},
        "comparisonData": {...}
    }

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask 예제

`backend/app.py`:

```python
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # CORS 활성화

@app.route('/api/merchants/<merchant_id>/home', methods=['GET'])
def get_home_data(merchant_id):
    """홈 화면 데이터"""
    # 실제로는 DB에서 데이터를 가져옴
    data = {
        "healthData": {
            "score": 72,
            "status": "warning",
            "daysUntilCrisis": 45
        },
        "riskSignals": [...]
    }
    return jsonify(data)

@app.route('/api/merchants/<merchant_id>/report', methods=['GET'])
def get_detail_report(merchant_id):
    """상세 리포트 데이터"""
    # CSV 파일 읽기 예제
    import pandas as pd
    
    # CSV에서 데이터 로드
    sales_df = pd.read_csv(f'data/sales_{merchant_id}.csv')
    
    data = {
        "salesData": sales_df.to_dict('records'),
        "customerData": {...},
        "comparisonData": {...}
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
```

---

## 💡 예제 코드

### 완전한 App.jsx 예제 (API 연동)

```javascript
import React, { useState, useEffect } from 'react';
import HomeScreen from './screens/HomeScreen';
import DetailReportScreen from './screens/DetailReportScreen';
import SolutionScreen from './screens/SolutionScreen';
import WebDashboardScreen from './screens/WebDashboardScreen';
import AdminConsoleScreen from './screens/AdminConsoleScreen';
import { api } from './services/api';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const merchantId = '12345'; // 실제로는 로그인에서 가져옴

  // 화면별 데이터 로드
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        let result;
        switch (currentScreen) {
          case 'home':
            result = await api.getHomeData(merchantId);
            break;
          case 'detail':
            result = await api.getDetailReport(merchantId);
            break;
          case 'solution':
            result = await api.getSolutions(merchantId);
            break;
          case 'dashboard':
            result = await api.getDashboard(merchantId);
            break;
          case 'admin':
            result = await api.getAdminData('성동구');
            break;
          default:
            result = await api.getHomeData(merchantId);
        }
        
        setData(result);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [currentScreen, merchantId]);

  // 로딩 상태
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-xl font-bold text-blue-700">로딩 중...</div>
        </div>
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-xl font-bold text-red-700">에러 발생</div>
          <div className="text-gray-600 mt-2">{error}</div>
          <button 
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  // 화면 렌더링
  const navigate = (screen) => setCurrentScreen(screen);

  switch (currentScreen) {
    case 'home':
      return (
        <HomeScreen
          {...data}
          onDetailReport={() => navigate('detail')}
          onSolutionCheck={() => navigate('solution')}
          onNavigate={navigate}
        />
      );
    
    case 'detail':
      return (
        <DetailReportScreen
          {...data}
          onBack={() => navigate('home')}
        />
      );
    
    case 'solution':
      return (
        <SolutionScreen
          {...data}
          onBack={() => navigate('home')}
        />
      );
    
    case 'dashboard':
      return (
        <WebDashboardScreen
          {...data}
          onNavigate={navigate}
        />
      );
    
    case 'admin':
      return (
        <AdminConsoleScreen
          {...data}
        />
      );
    
    default:
      return <HomeScreen {...data} onNavigate={navigate} />;
  }
}
```

---

## 🔧 환경 변수 설정

`.env` 파일 생성:

```bash
# API 서버 주소
REACT_APP_API_URL=http://localhost:8000

# 개발/프로덕션 환경
REACT_APP_ENV=development
```

---

## 📝 체크리스트

### 백엔드 개발자가 준비할 것:

- [ ] API 엔드포인트 구현
- [ ] CORS 설정
- [ ] 인증/인가 구현 (필요시)
- [ ] 데이터 검증 (Pydantic, Marshmallow 등)
- [ ] 에러 처리
- [ ] API 문서 (Swagger/OpenAPI)

### 프론트엔드 개발자가 준비할 것:

- [ ] `src/services/api.js` 생성
- [ ] 에러 핸들링 UI
- [ ] 로딩 상태 UI
- [ ] 환경 변수 설정
- [ ] API 응답 데이터 형식 확인

---

## 🚀 빠른 시작

### 1. 백엔드 실행 (Python)

```bash
cd backend
pip install fastapi uvicorn
uvicorn main:app --reload
```

### 2. 프론트엔드 실행 (React)

```bash
cd shinhan-healthcare
npm start
```

### 3. API 테스트

브라우저에서 `http://localhost:8000/docs` 접속 (FastAPI의 경우)

---

## 📞 문의

- 프론트엔드 관련: React 개발팀
- 백엔드 API 관련: Python 개발팀
- 데이터 형식 문의: 데이터 엔지니어링팀