import React, { useState, useEffect } from 'react';

// 각 화면 컴포넌트 import
import HomeScreen from './screens/HomeScreen';
import DetailReportScreen from './screens/DetailReportScreen';
import SolutionScreen from './screens/SolutionScreen';
import WebDashboardScreen from './screens/WebDashboardScreen';
import AdminConsoleScreen from './screens/AdminConsoleScreen';

// API service
import {
  getMerchantRisk,
  getMerchantSignals,
  getMerchantSales,
  getRegionOverview,
  listMerchants,
  transformRiskDataForHome,
  transformSalesDataForDetail,
  transformDataForDashboard,
  transformDataForAdmin,
} from './services/api';

// Solution 데이터는 정적으로 유지
const SOLUTION_DATA = {
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
};

export default function AppWithAPI() {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [currentMerchant, setCurrentMerchant] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Data states
  const [homeData, setHomeData] = useState(null);
  const [detailData, setDetailData] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [adminData, setAdminData] = useState(null);

  // Load sample merchant on mount
  useEffect(() => {
    loadSampleMerchant();
  }, []);

  /**
   * Load a sample merchant for demonstration
   */
  const loadSampleMerchant = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get list of merchants
      const merchantList = await listMerchants();
      if (merchantList.merchants && merchantList.merchants.length > 0) {
        // Use first merchant
        const sampleMerchantId = merchantList.merchants[0];
        setCurrentMerchant(sampleMerchantId);
        await loadMerchantData(sampleMerchantId);
      } else {
        setError('No merchants found in database');
      }
    } catch (err) {
      console.error('Error loading sample merchant:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Load data for a specific merchant
   */
  const loadMerchantData = async (merchantId) => {
    try {
      setLoading(true);
      setError(null);

      // Load risk, signals, and sales data
      const [riskData, signalsData, salesData] = await Promise.all([
        getMerchantRisk(merchantId),
        getMerchantSignals(merchantId),
        getMerchantSales(merchantId),
      ]);

      // Transform for different screens
      setHomeData(transformRiskDataForHome(riskData, signalsData));
      setDetailData(transformSalesDataForDetail(salesData, riskData));
      setDashboardData(transformDataForDashboard(riskData, salesData, signalsData));

      console.log('✅ Merchant data loaded:', merchantId);
    } catch (err) {
      console.error('Error loading merchant data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Load admin console data
   */
  const loadAdminData = async () => {
    try {
      setLoading(true);
      setError(null);

      const overviewData = await getRegionOverview();
      setAdminData(transformDataForAdmin(overviewData));

      console.log('✅ Admin data loaded');
    } catch (err) {
      console.error('Error loading admin data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // 화면 전환 함수
  const navigate = (screen) => {
    setCurrentScreen(screen);
    console.log('Navigating to:', screen);

    // Load admin data when navigating to admin screen
    if (screen === 'admin' && !adminData) {
      loadAdminData();
    }
  };

  // 이벤트 핸들러들
  const handleLogout = () => {
    console.log('Logout clicked');
    navigate('home');
  };

  const handleDetailClick = (item) => {
    console.log('Detail clicked:', item);
    // If item has merchant_id, load that merchant's data
    if (item && item.merchant_id) {
      setCurrentMerchant(item.merchant_id);
      loadMerchantData(item.merchant_id).then(() => {
        navigate('dashboard');
      });
    }
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

  // Loading state
  if (loading && !homeData) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">데이터 로딩 중...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error && !homeData) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center max-w-md mx-auto p-6">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">데이터 로딩 실패</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={loadSampleMerchant}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  // 화면 렌더링
  switch (currentScreen) {
    case 'home':
      return homeData ? (
        <HomeScreen
          healthData={homeData.healthData}
          riskSignals={homeData.riskSignals}
          onDetailReport={() => navigate('detail')}
          onSolutionCheck={() => navigate('solution')}
          onNavigate={navigate}
        />
      ) : null;

    case 'detail':
      return detailData ? (
        <DetailReportScreen
          salesData={detailData.salesData}
          customerData={detailData.customerData}
          comparisonData={detailData.comparisonData}
          onBack={() => navigate('home')}
        />
      ) : null;

    case 'solution':
      return (
        <SolutionScreen
          improvementSolutions={SOLUTION_DATA.improvementSolutions}
          financialProducts={SOLUTION_DATA.financialProducts}
          onBack={() => navigate('home')}
          onDetailClick={handleDetailClick}
          onConsultClick={handleConsultClick}
        />
      );

    case 'dashboard':
      return dashboardData ? (
        <WebDashboardScreen
          healthData={dashboardData.healthData}
          salesData={dashboardData.salesData}
          customerData={dashboardData.customerData}
          shapData={dashboardData.shapData}
          onNavigate={navigate}
          onLogout={handleLogout}
          onSolutionClick={() => navigate('solution')}
          onConsultClick={handleConsultClick}
        />
      ) : null;

    case 'admin':
      return adminData ? (
        <AdminConsoleScreen
          regionName={adminData.regionName}
          totalStores={adminData.totalStores}
          statusBreakdown={adminData.statusBreakdown}
          riskMerchants={adminData.riskMerchants}
          onLogout={handleLogout}
          onDetailClick={handleDetailClick}
          onBulkNotify={handleBulkNotify}
          onAssignExpert={handleAssignExpert}
        />
      ) : (
        <div className="flex items-center justify-center h-screen">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      );

    default:
      return homeData ? (
        <HomeScreen
          healthData={homeData.healthData}
          riskSignals={homeData.riskSignals}
          onDetailReport={() => navigate('detail')}
          onSolutionCheck={() => navigate('solution')}
          onNavigate={navigate}
        />
      ) : null;
  }
}
