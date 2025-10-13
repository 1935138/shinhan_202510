import React from 'react';
import { ArrowLeft, TrendingUp, Users, Trophy } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// 헤더 컴포넌트
const ReportHeader = ({ onBack }) => {
  return (
    <div className="bg-white px-5 py-4 flex items-center gap-3 sticky top-0 z-10 shadow-md">
      <button onClick={onBack}>
        <ArrowLeft className="w-6 h-6 text-blue-700" />
      </button>
      <h1 className="text-lg font-bold text-blue-700">대시보드</h1>
    </div>
  );
};

// 섹션 헤더 컴포넌트
const SectionHeader = ({ icon: Icon, title, color = 'blue' }) => {
  const colorClasses = {
    blue: 'text-blue-700',
    green: 'text-green-700',
    purple: 'text-purple-700',
  };

  return (
    <div className="flex items-center gap-2 mb-3">
      <Icon className={`w-5 h-5 ${colorClasses[color]}`} />
      <h2 className="text-base font-bold text-gray-900">{title}</h2>
    </div>
  );
};

// 매출 추이 컴포넌트
const SalesTrendChart = ({ data }) => {
  return (
    <div className="bg-white rounded-2xl shadow-md p-5 border border-gray-100">
      <SectionHeader icon={TrendingUp} title="매출 추이 (6개월)" color="blue" />
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="month" 
              tick={{ fontSize: 12 }}
              stroke="#666"
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              stroke="#666"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px'
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line 
              type="monotone" 
              dataKey="sales" 
              stroke="#1d4ed8" 
              strokeWidth={3}
              name="매출 (만원)"
              dot={{ fill: '#1d4ed8', r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// 고객 분석 컴포넌트
const CustomerAnalysis = ({ data }) => {
  const pieData = [
    { name: '재방문 고객', value: data.returning, color: '#1d4ed8' },
    { name: '신규 고객', value: data.new, color: '#93c5fd' },
  ];

  const ageData = data.ageDistribution || [];

  return (
    <div className="bg-white rounded-2xl shadow-md p-5 border border-gray-100">
      <SectionHeader icon={Users} title="고객 분석" color="green" />
      
      {/* 고객 비율 */}
      <div className="grid grid-cols-2 gap-3 mb-5">
        <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
          <p className="text-xs text-gray-600 mb-1">재방문 고객</p>
          <p className="text-2xl font-bold text-blue-700">{data.returning}%</p>
        </div>
        <div className="bg-sky-50 rounded-xl p-4 border border-sky-100">
          <p className="text-xs text-gray-600 mb-1">신규 고객</p>
          <p className="text-2xl font-bold text-sky-700">{data.new}%</p>
        </div>
      </div>

      {/* 고객층 분포 차트 */}
      <div className="h-64">
        <p className="text-sm font-semibold text-gray-700 mb-3">연령대별 분포</p>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={ageData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="age" 
              tick={{ fontSize: 12 }}
              stroke="#666"
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              stroke="#666"
              label={{ value: '비율 (%)', angle: -90, position: 'insideLeft', fontSize: 12 }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px'
              }}
            />
            <Bar dataKey="percentage" fill="#10b981" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// 업종/상권 비교 컴포넌트
const IndustryComparison = ({ data }) => {
  return (
    <div className="bg-white rounded-2xl shadow-md p-5 border border-gray-100">
      <SectionHeader icon={Trophy} title="업종/상권 비교" color="purple" />
      
      <div className="space-y-4">
        {/* 업종 평균 대비 */}
        <div className="bg-purple-50 rounded-xl p-4 border border-purple-100">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium text-gray-700">업종 평균 대비</p>
            <p className="text-2xl font-bold text-purple-700">{data.industryAverage}%</p>
          </div>
          <div className="w-full bg-purple-200 rounded-full h-3">
            <div 
              className="bg-purple-600 h-3 rounded-full transition-all"
              style={{ width: `${data.industryAverage}%` }}
            />
          </div>
        </div>

        {/* 상권 내 순위 */}
        <div className="bg-indigo-50 rounded-xl p-4 border border-indigo-100">
          <p className="text-sm font-medium text-gray-700 mb-2">상권 내 순위</p>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-bold text-indigo-700">{data.ranking}</p>
            <p className="text-lg text-gray-600">/ {data.totalStores}개</p>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            상위 {Math.round((data.ranking / data.totalStores) * 100)}%
          </p>
        </div>

        {/* 추가 지표 */}
        {data.additionalMetrics && (
          <div className="grid grid-cols-2 gap-3 mt-4">
            {data.additionalMetrics.map((metric, index) => (
              <div key={index} className="bg-gray-50 rounded-xl p-3 border border-gray-200">
                <p className="text-xs text-gray-600 mb-1">{metric.label}</p>
                <p className="text-lg font-bold text-gray-900">{metric.value}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// 메인 상세 리포트 화면
export default function DetailReportScreen({
  salesData = [
    { month: '4월', sales: 3200 },
    { month: '5월', sales: 3800 },
    { month: '6월', sales: 3500 },
    { month: '7월', sales: 4200 },
    { month: '8월', sales: 3900 },
    { month: '9월', sales: 3600 },
  ],
  customerData = {
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
  comparisonData = {
    industryAverage: 85,
    ranking: 15,
    totalStores: 50,
    additionalMetrics: [
      { label: '전월 대비', value: '-5%' },
      { label: '전년 대비', value: '+12%' },
    ],
  },
  onBack,
}) {
  return (
    <div className="min-h-screen bg-gray-50 pb-8">
      {/* Header */}
      <ReportHeader onBack={onBack} />

      {/* Content */}
      <div className="px-5 pt-5 space-y-5">
        {/* 매출 추이 */}
        <SalesTrendChart data={salesData} />

        {/* 고객 분석 */}
        <CustomerAnalysis data={customerData} />

        {/* 업종/상권 비교 */}
        <IndustryComparison data={comparisonData} />
      </div>
    </div>
  );
}