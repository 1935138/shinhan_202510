import React from 'react';
import { BarChart3, FileText, Settings, LogOut, AlertTriangle, TrendingUp, Users, Lightbulb, MessageSquare } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// 상단 네비게이션 바
const TopNavigation = ({ onNavigate, onLogout }) => {
  const navItems = [
    { id: 'dashboard', label: '대시보드', icon: BarChart3 },
    { id: 'report', label: '리포트', icon: FileText },
    { id: 'settings', label: '설정', icon: Settings },
  ];

  return (
    <div className="bg-white shadow-md px-8 py-4 flex items-center justify-between sticky top-0 z-10">
      <div className="flex items-center gap-8">
        <h1 className="text-xl font-bold text-blue-700">신한 가맹점 헬스케어</h1>
        
        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors text-gray-700 hover:text-blue-700"
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium text-sm">{item.label}</span>
              </button>
            );
          })}
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

// 건강 점수 카드
const HealthScoreCard = ({ score = 72, status = 'warning', daysUntilCrisis = 45 }) => {
  const statusConfig = {
    warning: {
      label: '경고',
      color: 'text-amber-700',
      bgColor: 'bg-amber-50',
      borderColor: 'border-amber-300',
    },
    danger: {
      label: '위험',
      color: 'text-red-700',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-300',
    },
    good: {
      label: '양호',
      color: 'text-green-700',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-300',
    },
  };

  const config = statusConfig[status] || statusConfig.warning;

  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
      <h3 className="text-sm font-semibold text-gray-600 mb-4">건강 점수</h3>
      <div className="flex flex-col items-center">
        <div className="text-5xl font-bold text-gray-900 mb-3">{score}<span className="text-2xl text-gray-500">점</span></div>
        <div className={`flex items-center gap-2 ${config.bgColor} ${config.borderColor} border-2 rounded-full px-4 py-2`}>
          <AlertTriangle className={`w-4 h-4 ${config.color}`} />
          <span className={`${config.color} font-bold text-sm`}>{config.label}</span>
        </div>
        <p className="text-xs text-gray-500 mt-3">예상 위기: {daysUntilCrisis}일 후</p>
      </div>
    </div>
  );
};

// 매출 추이 카드
const SalesTrendCard = ({ data }) => {
  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
      <h3 className="text-sm font-semibold text-gray-600 mb-4">매출 추이</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} stroke="#666" />
            <YAxis tick={{ fontSize: 11 }} stroke="#666" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '11px'
              }}
            />
            <Line
              type="monotone"
              dataKey="sales"
              stroke="#1d4ed8"
              strokeWidth={2}
              dot={{ fill: '#1d4ed8', r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// 고객 분석 카드
const CustomerAnalysisCard = ({ data }) => {
  const pieData = [
    { name: '재방문', value: data.returning, color: '#1d4ed8' },
    { name: '신규', value: data.new, color: '#93c5fd' },
  ];

  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
      <h3 className="text-sm font-semibold text-gray-600 mb-4">고객 분석</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={40}
              outerRadius={70}
              paddingAngle={2}
              dataKey="value"
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// SHAP 분석 바 컴포넌트
const ShapBar = ({ label, value, maxValue = 0.3 }) => {
  const percentage = (value / maxValue) * 100;
  
  // 값에 따라 색상 그라데이션 (높을수록 진한 주황/빨강)
  const getColor = (val) => {
    if (val >= 0.2) return 'from-orange-400 to-red-500';
    if (val >= 0.15) return 'from-orange-300 to-orange-500';
    return 'from-yellow-400 to-orange-400';
  };
  
  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        <span className="text-sm font-bold text-orange-700">{value.toFixed(2)}</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className={`bg-gradient-to-r ${getColor(value)} h-3 rounded-full transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

// 위험 요인 분석 섹션
const RiskFactorAnalysis = ({ shapData }) => {
  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
      <h3 className="text-sm font-semibold text-gray-600 mb-4 flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-blue-700" />
        위험 요인 분석 - SHAP 분석 결과
      </h3>
      <div className="mt-4">
        {shapData.map((item, index) => (
          <ShapBar
            key={index}
            label={item.label}
            value={item.value}
          />
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-4">
        * SHAP 값이 높을수록 건강 점수에 부정적인 영향을 미치는 요인입니다.
      </p>
    </div>
  );
};

// 액션 카드 컴포넌트
const ActionCard = ({ icon: Icon, title, description, buttonText, onClick, color = 'blue' }) => {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100',
    green: 'bg-green-50 border-green-200 text-green-700 hover:bg-green-100',
  };

  return (
    <div className={`${colorClasses[color]} rounded-xl p-6 border-2 transition-all hover:shadow-md`}>
      <div className="flex items-start gap-3 mb-4">
        <div className={`w-10 h-10 rounded-full ${color === 'blue' ? 'bg-blue-200' : 'bg-green-200'} flex items-center justify-center`}>
          <Icon className={`w-5 h-5 ${color === 'blue' ? 'text-blue-700' : 'text-green-700'}`} />
        </div>
        <div className="flex-1">
          <h4 className="font-bold text-gray-900 mb-1">{title}</h4>
          <p className="text-sm text-gray-600">{description}</p>
        </div>
      </div>
      <button
        onClick={onClick}
        className={`w-full ${color === 'blue' ? 'bg-blue-700 hover:bg-blue-800' : 'bg-green-700 hover:bg-green-800'} text-white font-semibold py-3 rounded-lg transition-colors`}
      >
        {buttonText}
      </button>
    </div>
  );
};

// 메인 웹 대시보드
export default function WebDashboardScreen({
  healthData = {
    score: 72,
    status: 'warning',
    daysUntilCrisis: 45,
  },
  salesData = [
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
  customerData = {
    returning: 45,
    new: 55,
  },
  shapData = [
    { label: '재방문율 감소', value: 0.25 },
    { label: '업종 순위 하락', value: 0.18 },
    { label: '신규 고객 유입 감소', value: 0.15 },
    { label: '평균 객단가 하락', value: 0.12 },
    { label: '배달 의존도 증가', value: 0.10 },
  ],
  onNavigate,
  onLogout,
  onSolutionClick,
  onConsultClick,
}) {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <TopNavigation onNavigate={onNavigate} onLogout={onLogout} />

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-6">
        {/* Top Cards Row */}
        <div className="grid grid-cols-3 gap-6 mb-6">
          <HealthScoreCard
            score={healthData.score}
            status={healthData.status}
            daysUntilCrisis={healthData.daysUntilCrisis}
          />
          <SalesTrendCard data={salesData} />
          <CustomerAnalysisCard data={customerData} />
        </div>

        {/* Risk Factor Analysis */}
        <div className="mb-6">
          <RiskFactorAnalysis shapData={shapData} />
        </div>

        {/* Action Cards */}
        <div className="grid grid-cols-2 gap-6">
          <ActionCard
            icon={Lightbulb}
            title="추천 솔루션"
            description="AI가 분석한 맞춤형 개선 방안을 확인하세요"
            buttonText="솔루션 보기"
            onClick={onSolutionClick}
            color="blue"
          />
          <ActionCard
            icon={MessageSquare}
            title="전문가 상담"
            description="신한카드 전문 컨설턴트와 1:1 상담하세요"
            buttonText="상담 신청하기"
            onClick={onConsultClick}
            color="green"
          />
        </div>
      </div>
    </div>
  );
}