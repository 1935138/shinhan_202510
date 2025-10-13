import React from 'react';
import { ArrowLeft, Target, DollarSign, TrendingUp, Clock, ArrowRight } from 'lucide-react';

// 헤더 컴포넌트
const SolutionHeader = ({ onBack }) => {
  return (
    <div className="bg-white px-5 py-4 flex items-center gap-3 sticky top-0 z-10 shadow-md">
      <button onClick={onBack}>
        <ArrowLeft className="w-6 h-6 text-blue-700" />
      </button>
      <h1 className="text-lg font-bold text-blue-700">추천 솔루션</h1>
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
    <div className="flex items-center gap-2 mb-4">
      <Icon className={`w-5 h-5 ${colorClasses[color]}`} />
      <h2 className="text-base font-bold text-gray-900">{title}</h2>
    </div>
  );
};

// 개선안 카드 컴포넌트
const ImprovementCard = ({ solution, onDetailClick }) => {
  return (
    <div className="bg-white rounded-2xl shadow-md p-5 border border-gray-100 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-base font-bold text-gray-900 flex-1">
          {solution.priority}. {solution.title}
        </h3>
        {solution.badge && (
          <span className="bg-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded-full">
            {solution.badge}
          </span>
        )}
      </div>

      {solution.description && (
        <p className="text-sm text-gray-600 mb-4">{solution.description}</p>
      )}

      {/* 효과 및 비용 */}
      <div className="space-y-2 mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-green-600" />
          <span className="text-sm text-gray-700">
            <span className="font-medium">예상 효과:</span> {solution.expectedEffect}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-blue-600" />
          <span className="text-sm text-gray-700">
            <span className="font-medium">소요 비용:</span> {solution.cost}
          </span>
        </div>

        {solution.duration && (
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-purple-600" />
            <span className="text-sm text-gray-700">
              <span className="font-medium">예상 기간:</span> {solution.duration}
            </span>
          </div>
        )}
      </div>

      {/* 자세히 보기 버튼 */}
      <button 
        onClick={() => onDetailClick(solution)}
        className="w-full bg-blue-50 text-blue-700 font-semibold py-3 rounded-xl hover:bg-blue-100 transition-colors flex items-center justify-center gap-2"
      >
        자세히 보기
        <ArrowRight className="w-4 h-4" />
      </button>
    </div>
  );
};

// 금융상품 카드 컴포넌트
const FinancialProductCard = ({ product, onConsultClick }) => {
  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-md p-5 border border-blue-100">
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-base font-bold text-gray-900">{product.name}</h3>
        {product.badge && (
          <span className="bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded-full">
            {product.badge}
          </span>
        )}
      </div>

      {product.description && (
        <p className="text-sm text-gray-600 mb-4">{product.description}</p>
      )}

      {/* 상품 정보 */}
      <div className="bg-white rounded-xl p-4 mb-4 space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">금리</span>
          <span className="text-lg font-bold text-blue-700">{product.interestRate}</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">한도</span>
          <span className="text-lg font-bold text-gray-900">{product.maxAmount}</span>
        </div>

        {product.additionalInfo && product.additionalInfo.map((info, index) => (
          <div key={index} className="flex justify-between items-center">
            <span className="text-sm text-gray-600">{info.label}</span>
            <span className="text-sm font-semibold text-gray-900">{info.value}</span>
          </div>
        ))}
      </div>

      {/* 주요 혜택 */}
      {product.benefits && product.benefits.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-semibold text-gray-700 mb-2">주요 혜택</p>
          <ul className="space-y-1">
            {product.benefits.map((benefit, index) => (
              <li key={index} className="text-xs text-gray-600 flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">•</span>
                <span>{benefit}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 상담 신청 버튼 */}
      <button 
        onClick={() => onConsultClick(product)}
        className="w-full bg-blue-700 text-white font-bold py-3 rounded-xl hover:bg-blue-800 transition-colors shadow-md"
      >
        상담 신청
      </button>
    </div>
  );
};

// 메인 추천 솔루션 화면
export default function SolutionScreen({
  improvementSolutions = [
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
  financialProducts = [
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
    {
      name: '운영자금 대출',
      description: '가맹점 운영 자금 지원을 위한 대출',
      interestRate: '4.2%',
      maxAmount: '최대 1,000만원',
      additionalInfo: [
        { label: '상환 기간', value: '최대 24개월' },
      ],
      benefits: [
        '신한카드 가맹점 우대 금리',
        '서류 간소화',
      ],
    },
  ],
  onBack,
  onDetailClick,
  onConsultClick,
}) {
  return (
    <div className="min-h-screen bg-gray-50 pb-8">
      {/* Header */}
      <SolutionHeader onBack={onBack} />

      {/* Content */}
      <div className="px-5 pt-5 space-y-8">
        {/* 즉시 실행 가능한 개선안 */}
        <section className="bg-blue-50 rounded-3xl p-5 border-2 border-blue-200">
          <SectionHeader icon={Target} title="즉시 실행 가능한 개선안" color="blue" />
          <div className="space-y-4">
            {improvementSolutions.map((solution, index) => (
              <ImprovementCard
                key={index}
                solution={solution}
                onDetailClick={onDetailClick}
              />
            ))}
          </div>
        </section>

        {/* 구분선 */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-gray-300"></div>
          <span className="text-sm font-semibold text-gray-500">금융 지원</span>
          <div className="flex-1 h-px bg-gray-300"></div>
        </div>

        {/* 추천 금융상품 */}
        <section className="bg-green-50 rounded-3xl p-5 border-2 border-green-200">
          <SectionHeader icon={DollarSign} title="추천 금융상품" color="green" />
          <div className="space-y-4">
            {financialProducts.map((product, index) => (
              <FinancialProductCard
                key={index}
                product={product}
                onConsultClick={onConsultClick}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}