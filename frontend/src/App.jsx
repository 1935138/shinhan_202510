import React, { useState } from 'react';

// 각 화면 컴포넌트 import
import HomeScreen from './UX/HomeScreen';
import DetailReportScreen from './UX/DetailReportScreen';
import SolutionScreen from './UX/SolutionScreen';
import WebDashboardScreen from './UX/WebDashboardScreen';
import AdminConsoleScreen from './UX/AdminConsoleScreen';

// 샘플 데이터 (나중에 JSON 파일에서 로드)
const SAMPLE_DATA = {
  home: {
    healthData: {
      score: 72,
      status: 'warning',
      daysUntilCrisis: 45,
    },
    riskSignals: [
      {
        icon: 'repeat',
        title: '재방문 고객 20% 감소',
        description: '지난 달 대비 재방문율 급격히 하락',
        color: 'red',
      },
      {
        icon: 'trending',
        title: '업종 내 순위 15% 하락',
        description: '주변 경쟁 가맹점 대비 성과 저조',
        color: 'orange',
      },
      {
        icon: 'package',
        title: '배달 의존도 증가',
        description: '내점 매출 감소, 배달 비중 상승',
        color: 'amber',
      },
    ],
  },

  detailReport: {
    salesData: [
      { month: '4월', sales: 3200 },
      { month: '5월', sales: 3800 },
      { month: '6월', sales: 3500 },
      { month: '7월', sales: 4200 },
      { month: '8월', sales: 3900 },
      { month: '9월', sales: 3600 },
    ],
    customerData: {
      returning: 45,
      new: 55,
      ageDistribution: [
        { age: '20대', percentage: 15 },
        { age: '30대', percentage: 35 },
        { age: '40대', percentage: 30 },
        { age: '50대', percentage: 15 },
        { age: '60대+', percentage: 5 },
      ],
    },
    comparisonData: {
      industryAverage: 85,
      ranking: 15,
      totalStores: 50,
      additionalMetrics: [
        { label: '전월 대비', value: '-5%' },
        { label: '전년 대비', value: '+12%' },
      ],
    },
  },

  solution: {
    improvementSolutions: [
      {
        priority: 1,
        title: '고객 재방문 캠페인',
        description: '기존 고객 대상 맞춤형 할인 쿠폰 제공',
        expectedEffect: '재방문 15% ↑',
        cost: '30만원',
        duration: '1개월',
        badge: '긴급',
      },
      {
        priority: 2,
        title: '배달앱 프로모션',
        description: '배달 주문 고객 대상 할인 이벤트',
        expectedEffect: '배달 매출 20% ↑',
        cost: '50만원',
        duration: '2주',
      },
      {
        priority: 3,
        title: 'SNS 마케팅 강화',
        description: '인스타그램, 블로그 홍보 캠페인',
        expectedEffect: '신규 고객 25% ↑',
        cost: '40만원',
        duration: '1개월',
      },
    ],
    financialProducts: [
      {
        name: '마케팅 지원 대출',
        description: '가맹점 마케팅 활동을 위한 특별 금리 대출',
        interestRate: '3.5%',
        maxAmount: '최대 500만원',
        badge: '추천',
        additionalInfo: [
          { label: '상환 기간', value: '최대 12개월' },
          { label: '대출 기간', value: '즉시 가능' },
        ],
        benefits: [
          '마케팅 비용 사용 시 금리 우대',
          '3개월 거치 가능',
          '중도상환 수수료 면제',
        ],
      },
    ],
  },

  webDashboard: {
    healthData: {
      score: 72,
      status: 'warning',
      daysUntilCrisis: 45,
    },
    salesData: [
      { month: '4월', sales: 3200 },
      { month: '5월', sales: 3800 },
      { month: '6월', sales: 3500 },
      { month: '7월', sales: 4200 },
      { month: '8월', sales: 3900 },
      { month: '9월', sales: 3600 },
      { month: '10월', sales: 3400 },
      { month: '11월', sales: 3700 },
      { month: '12월', sales: 4100 },
      { month: '1월', sales: 3300 },
      { month: '2월', sales: 3500 },
      { month: '3월', sales: 3800 },
    ],
    customerData: {
      returning: 45,
      new: 55,
    },
    shapData: [
      { label: '재방문율 감소', value: 0.25 },
      { label: '업종 순위 하락', value: 0.18 },
      { label: '신규 고객 유입 감소', value: 0.15 },
      { label: '평균 객단가 하락', value: 0.12 },
      { label: '배달 의존도 증가', value: 0.10 },
    ],
  },

  adminConsole: {
    regionName: '성동구',
    totalStores: 4185,
    statusBreakdown: [
      { status: '안전', count: 2500, percentage: 60, color: 'safe' },
      { status: '양호', count: 1000, percentage: 24, color: 'good' },
      { status: '주의', count: 500, percentage: 12, color: 'caution' },
      { status: '경고', count: 150, percentage: 3.5, color: 'warning' },
      { status: '위험', count: 35, percentage: 0.8, color: 'danger' },
    ],
    riskMerchants: [
      {
        name: '호호치킨',
        category: '치킨',
        riskScore: 92,
        riskType: '매출급락형',
        lastConsultDate: '2024-09-15',
        isUrgent: true,
      },
      {
        name: '카페***',
        category: '카페',
        riskScore: 85,
        riskType: '고객이탈형',
        lastConsultDate: '2024-09-20',
        isUrgent: false,
      },
      {
        name: '한식당**',
        category: '한식',
        riskScore: 78,
        riskType: '경쟁열위형',
        lastConsultDate: null,
        isUrgent: false,
      },
    ],
  },
};

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('home');

  // 화면 전환 함수
  const navigate = (screen) => {
    setCurrentScreen(screen);
    console.log('Navigating to:', screen);
  };

  // 이벤트 핸들러들
  const handleLogout = () => {
    console.log('Logout clicked');
    navigate('home');
  };

  const handleDetailClick = (item) => {
    console.log('Detail clicked:', item);
  };

  const handleConsultClick = (product) => {
    console.log('Consult clicked:', product);
  };

  const handleBulkNotify = () => {
    console.log('Bulk notify clicked');
  };

  const handleAssignExpert = () => {
    console.log('Assign expert clicked');
  };

  // 화면 렌더링
  switch (currentScreen) {
    case 'home':
      return (
        <HomeScreen
          healthData={SAMPLE_DATA.home.healthData}
          riskSignals={SAMPLE_DATA.home.riskSignals}
          onDetailReport={() => navigate('detail')}
          onSolutionCheck={() => navigate('solution')}
          onNavigate={navigate}
        />
      );

    case 'detail':
      return (
        <DetailReportScreen
          salesData={SAMPLE_DATA.detailReport.salesData}
          customerData={SAMPLE_DATA.detailReport.customerData}
          comparisonData={SAMPLE_DATA.detailReport.comparisonData}
          onBack={() => navigate('home')}
        />
      );

    case 'solution':
      return (
        <SolutionScreen
          improvementSolutions={SAMPLE_DATA.solution.improvementSolutions}
          financialProducts={SAMPLE_DATA.solution.financialProducts}
          onBack={() => navigate('home')}
          onDetailClick={handleDetailClick}
          onConsultClick={handleConsultClick}
        />
      );

    case 'dashboard':
      return (
        <WebDashboardScreen
          healthData={SAMPLE_DATA.webDashboard.healthData}
          salesData={SAMPLE_DATA.webDashboard.salesData}
          customerData={SAMPLE_DATA.webDashboard.customerData}
          shapData={SAMPLE_DATA.webDashboard.shapData}
          onNavigate={navigate}
          onLogout={handleLogout}
          onSolutionClick={() => navigate('solution')}
          onConsultClick={handleConsultClick}
        />
      );

    case 'admin':
      return (
        <AdminConsoleScreen
          regionName={SAMPLE_DATA.adminConsole.regionName}
          totalStores={SAMPLE_DATA.adminConsole.totalStores}
          statusBreakdown={SAMPLE_DATA.adminConsole.statusBreakdown}
          riskMerchants={SAMPLE_DATA.adminConsole.riskMerchants}
          onLogout={handleLogout}
          onDetailClick={handleDetailClick}
          onBulkNotify={handleBulkNotify}
          onAssignExpert={handleAssignExpert}
        />
      );

    default:
      return (
        <HomeScreen
          healthData={SAMPLE_DATA.home.healthData}
          riskSignals={SAMPLE_DATA.home.riskSignals}
          onDetailReport={() => navigate('detail')}
          onSolutionCheck={() => navigate('solution')}
          onNavigate={navigate}
        />
      );
  }
}