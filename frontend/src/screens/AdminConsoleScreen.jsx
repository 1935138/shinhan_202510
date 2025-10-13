import React, { useState } from 'react';
import { Shield, BarChart3, TrendingDown, Settings, LogOut, Search, Filter, Send, UserPlus, Eye, AlertCircle } from 'lucide-react';

// 상단 네비게이션 바
const AdminNavigation = ({ activeTab, onTabChange, onLogout }) => {
  const navItems = [
    { id: 'overview', label: '전체 현황' },
    { id: 'risk', label: '위험군 관리' },
    { id: 'performance', label: '성과 분석' },
    { id: 'settings', label: '설정' },
  ];

  return (
    <div className="bg-white shadow-md px-8 py-4 flex items-center justify-between sticky top-0 z-10">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <Shield className="w-6 h-6 text-blue-700" />
          <h1 className="text-xl font-bold text-blue-700">관리자 콘솔</h1>
        </div>
        
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`px-4 py-2 rounded-lg transition-colors font-medium text-sm ${
                activeTab === item.id
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <button
        onClick={onLogout}
        className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-red-50 transition-colors text-gray-700 hover:text-red-700"
      >
        <LogOut className="w-4 h-4" />
        <span className="font-medium text-sm">로그아웃</span>
      </button>
    </div>
  );
};

// 상태별 카드 컴포넌트
const StatusCard = ({ status, count, percentage, color }) => {
  const colorClasses = {
    safe: 'bg-green-50 border-green-300 text-green-700',
    good: 'bg-blue-50 border-blue-300 text-blue-700',
    caution: 'bg-yellow-50 border-yellow-300 text-yellow-700',
    warning: 'bg-orange-50 border-orange-300 text-orange-700',
    danger: 'bg-red-50 border-red-300 text-red-700',
  };

  return (
    <div className={`${colorClasses[color]} border-2 rounded-xl p-4 text-center transition-all hover:shadow-md`}>
      <p className="text-sm font-semibold mb-2">{status}</p>
      <p className="text-3xl font-bold mb-1">{count.toLocaleString()}</p>
      <p className="text-sm font-medium">{percentage}%</p>
    </div>
  );
};

// 전체 가맹점 현황
const OverviewSection = ({ regionName, totalStores, statusBreakdown }) => {
  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-gray-900">전체 가맹점 현황 ({regionName})</h2>
        <div className="text-sm text-gray-600">
          총 가맹점: <span className="font-bold text-blue-700">{totalStores.toLocaleString()}개</span>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4">
        {statusBreakdown.map((item, index) => (
          <StatusCard
            key={index}
            status={item.status}
            count={item.count}
            percentage={item.percentage}
            color={item.color}
          />
        ))}
      </div>
    </div>
  );
};

// 위험 유형 배지
const RiskTypeBadge = ({ type }) => {
  const typeConfig = {
    '매출급락형': 'bg-red-100 text-red-700',
    '고객이탈형': 'bg-orange-100 text-orange-700',
    '경쟁열위형': 'bg-amber-100 text-amber-700',
  };

  return (
    <span className={`${typeConfig[type] || 'bg-gray-100 text-gray-700'} px-2 py-1 rounded-full text-xs font-semibold`}>
      {type}
    </span>
  );
};

// 위험도 표시
const RiskScore = ({ score }) => {
  const getColor = (score) => {
    if (score >= 90) return 'text-red-700 font-bold';
    if (score >= 80) return 'text-orange-700 font-bold';
    return 'text-amber-700 font-bold';
  };

  return <span className={`${getColor(score)} text-sm`}>{score}점</span>;
};

// 위험군 테이블 행
const RiskMerchantRow = ({ merchant, onDetailClick }) => {
  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          {merchant.isUrgent && <AlertCircle className="w-4 h-4 text-red-500" />}
          <span className="font-medium text-gray-900">{merchant.name}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-gray-600 text-sm">{merchant.category}</td>
      <td className="px-4 py-3">
        <RiskScore score={merchant.riskScore} />
      </td>
      <td className="px-4 py-3">
        <RiskTypeBadge type={merchant.riskType} />
      </td>
      <td className="px-4 py-3 text-gray-600 text-sm">
        {merchant.lastConsultDate || <span className="text-red-600 font-semibold">미상담</span>}
      </td>
      <td className="px-4 py-3">
        <button
          onClick={() => onDetailClick(merchant)}
          className="text-blue-600 hover:text-blue-800 font-medium text-sm"
        >
          상세
        </button>
      </td>
    </tr>
  );
};

// 위험군 목록 테이블
const RiskMerchantTable = ({ merchants, onDetailClick }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  const filteredMerchants = merchants.filter(m => {
    const matchesSearch = m.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || m.riskType === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-gray-900">
            위험군 가맹점 목록 
            <span className="text-red-600 ml-2">({merchants.length}개)</span>
          </h2>
        </div>

        {/* 검색 및 필터 */}
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="가맹점명 검색..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 유형</option>
            <option value="매출급락형">매출급락형</option>
            <option value="고객이탈형">고객이탈형</option>
            <option value="경쟁열위형">경쟁열위형</option>
          </select>
        </div>
      </div>

      {/* 테이블 */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">가맹점명</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">업종</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">위험도</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">위험유형</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">최종상담일</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">액션</th>
            </tr>
          </thead>
          <tbody>
            {filteredMerchants.map((merchant, index) => (
              <RiskMerchantRow
                key={index}
                merchant={merchant}
                onDetailClick={onDetailClick}
              />
            ))}
          </tbody>
        </table>
      </div>

      {filteredMerchants.length === 0 && (
        <div className="p-8 text-center text-gray-500">
          검색 결과가 없습니다.
        </div>
      )}
    </div>
  );
};

// 액션 버튼 섹션
const ActionButtons = ({ onBulkNotify, onAssignExpert, selectedCount = 0 }) => {
  return (
    <div className="flex gap-3 mt-6">
      <button
        onClick={onBulkNotify}
        className="flex items-center gap-2 px-6 py-3 bg-blue-700 text-white font-semibold rounded-lg hover:bg-blue-800 transition-colors shadow-md"
      >
        <Send className="w-4 h-4" />
        일괄 알림 발송
        {selectedCount > 0 && <span className="ml-1">({selectedCount})</span>}
      </button>
      
      <button
        onClick={onAssignExpert}
        className="flex items-center gap-2 px-6 py-3 bg-green-700 text-white font-semibold rounded-lg hover:bg-green-800 transition-colors shadow-md"
      >
        <UserPlus className="w-4 h-4" />
        전문가 배정
      </button>
    </div>
  );
};

// 메인 관리자 콘솔
export default function AdminConsoleScreen({
  regionName = '성동구',
  totalStores = 4185,
  statusBreakdown = [
    { status: '안전', count: 2500, percentage: 60, color: 'safe' },
    { status: '양호', count: 1000, percentage: 24, color: 'good' },
    { status: '주의', count: 500, percentage: 12, color: 'caution' },
    { status: '경고', count: 150, percentage: 3.5, color: 'warning' },
    { status: '위험', count: 35, percentage: 0.8, color: 'danger' },
  ],
  riskMerchants = [
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
  onLogout,
  onDetailClick,
  onBulkNotify,
  onAssignExpert,
}) {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <AdminNavigation
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onLogout={onLogout}
      />

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-6">
        {/* Overview Section */}
        <OverviewSection
          regionName={regionName}
          totalStores={totalStores}
          statusBreakdown={statusBreakdown}
        />

        {/* Risk Merchant Table */}
        <RiskMerchantTable
          merchants={riskMerchants}
          onDetailClick={onDetailClick}
        />

        {/* Action Buttons */}
        <ActionButtons
          onBulkNotify={onBulkNotify}
          onAssignExpert={onAssignExpert}
        />
      </div>
    </div>
  );
}