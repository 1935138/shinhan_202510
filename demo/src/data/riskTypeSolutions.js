// 위험 유형별 맞춤 솔루션 데이터

export const riskTypeSolutions = {
  '매출 급락형': {
    description: '최근 매출이 급격하게 감소하고 있는 상태입니다. 즉각적인 매출 회복 전략이 필요합니다.',
    severity: 'high',
    icon: '📉',
    actions: [
      {
        title: '긴급 프로모션 진행',
        description: '단골 고객 대상 할인 쿠폰 및 신규 고객 유치 이벤트',
        expectedEffect: '매출 15-20% 회복',
        cost: '30-50만원',
        duration: '2주',
        priority: 'critical'
      },
      {
        title: '메뉴 리뉴얼 및 가격 조정',
        description: '인기 메뉴 재구성 및 가격 경쟁력 강화',
        expectedEffect: '객단가 8-12% 상승',
        cost: '50-100만원',
        duration: '1개월',
        priority: 'high'
      },
      {
        title: '배달앱 최적화',
        description: '배달 플랫폼 프로모션 참여 및 노출 순위 개선',
        expectedEffect: '신규 고객 30% 증가',
        cost: '20-40만원',
        duration: '1주',
        priority: 'high'
      }
    ],
    financialProducts: [
      {
        name: '긴급 운영자금 대출',
        description: '매출 회복을 위한 마케팅 및 프로모션 자금',
        rate: '3.5%',
        limit: '최대 500만원',
        benefit: '3개월 거치, 최대 12개월 분할 상환'
      },
      {
        name: '가맹점 마케팅 지원 프로그램',
        description: '온라인/오프라인 통합 마케팅 지원',
        rate: '무료',
        limit: '컨설팅 포함',
        benefit: '전문 마케터 1:1 매칭'
      }
    ]
  },

  '고객 이탈형': {
    description: '재방문 고객이 줄어들고 단골 고객이 이탈하는 상황입니다. 고객 관계 회복이 시급합니다.',
    severity: 'high',
    icon: '👥',
    actions: [
      {
        title: '단골 고객 재방문 캠페인',
        description: 'VIP 고객 대상 특별 혜택 및 멤버십 프로그램',
        expectedEffect: '재방문율 15-25% 증가',
        cost: '30만원',
        duration: '2주',
        priority: 'critical'
      },
      {
        title: '고객 피드백 시스템 구축',
        description: '만족도 조사 및 개선사항 즉시 반영',
        expectedEffect: '고객 만족도 20% 향상',
        cost: '10만원',
        duration: '1개월',
        priority: 'high'
      },
      {
        title: '포인트/스탬프 적립 프로그램',
        description: '재방문 유도를 위한 리워드 시스템',
        expectedEffect: '재방문율 10% 증가',
        cost: '20만원',
        duration: '지속',
        priority: 'medium'
      }
    ],
    financialProducts: [
      {
        name: 'CRM 시스템 구축 지원',
        description: '고객 관리 시스템 도입 자금',
        rate: '3.0%',
        limit: '최대 300만원',
        benefit: 'CRM 전문가 컨설팅 포함'
      },
      {
        name: '고객 로열티 프로그램 패키지',
        description: '신한카드 포인트 연동 멤버십',
        rate: '무료',
        limit: '시스템 무료 제공',
        benefit: '신한 포인트 적립 혜택'
      }
    ]
  },

  '배달 의존형': {
    description: '배달 매출 비중이 과도하게 높아 수익성이 악화되고 있습니다. 매장 방문 고객 확보가 필요합니다.',
    severity: 'medium',
    icon: '🚚',
    actions: [
      {
        title: '매장 방문 유도 프로모션',
        description: '매장 방문 시 특별 할인 및 무료 음료 제공',
        expectedEffect: '매장 방문 고객 20% 증가',
        cost: '25만원',
        duration: '1개월',
        priority: 'high'
      },
      {
        title: '배달 수수료 절감 전략',
        description: '직접 배달 시스템 도입 또는 플랫폼 수수료 협상',
        expectedEffect: '수익률 5-8% 개선',
        cost: '50만원',
        duration: '2개월',
        priority: 'high'
      },
      {
        title: '테이크아웃 전용 메뉴 개발',
        description: '포장 최적화 메뉴로 수익성 개선',
        expectedEffect: '객단가 10% 증가',
        cost: '30만원',
        duration: '2주',
        priority: 'medium'
      }
    ],
    financialProducts: [
      {
        name: '매장 개선 지원 대출',
        description: '인테리어 개선 및 분위기 향상 자금',
        rate: '3.8%',
        limit: '최대 1,000만원',
        benefit: '6개월 거치, 36개월 분할'
      },
      {
        name: '직접 배달 시스템 구축 지원',
        description: '배달 인프라 구축 비용 지원',
        rate: '4.0%',
        limit: '최대 300만원',
        benefit: '장비 구매 포함'
      }
    ]
  },

  '경쟁 열위형': {
    description: '동일 상권 내 경쟁에서 밀리고 있는 상황입니다. 차별화 전략이 필요합니다.',
    severity: 'medium',
    icon: '🏆',
    actions: [
      {
        title: '차별화 메뉴 개발',
        description: '시그니처 메뉴 개발 및 독점 레시피 강화',
        expectedEffect: '고객 만족도 15% 향상',
        cost: '80만원',
        duration: '1개월',
        priority: 'critical'
      },
      {
        title: '브랜딩 및 SNS 마케팅',
        description: '인스타그램, 블로그 등 온라인 홍보 강화',
        expectedEffect: '신규 고객 25% 증가',
        cost: '40만원',
        duration: '지속',
        priority: 'high'
      },
      {
        title: '서비스 품질 개선',
        description: '직원 교육 및 고객 응대 수준 향상',
        expectedEffect: '재방문율 10% 증가',
        cost: '20만원',
        duration: '2주',
        priority: 'medium'
      }
    ],
    financialProducts: [
      {
        name: '메뉴 개발 지원 프로그램',
        description: '전문 셰프 컨설팅 및 레시피 개발 지원',
        rate: '무료',
        limit: '컨설팅 3회',
        benefit: '푸드 스타일리스트 포함'
      },
      {
        name: '브랜딩 컨설팅 패키지',
        description: 'BI/CI 개선 및 마케팅 전략 수립',
        rate: '3.5%',
        limit: '최대 500만원',
        benefit: '디자인 에이전시 연결'
      }
    ]
  },

  '종합 위기형': {
    description: '매출, 고객, 경쟁력 등 복합적인 문제가 발생한 위기 상황입니다. 전문가 개입이 반드시 필요합니다.',
    severity: 'critical',
    icon: '🚨',
    actions: [
      {
        title: '긴급 경영 컨설팅',
        description: '전문 컨설턴트의 종합 진단 및 턴어라운드 전략 수립',
        expectedEffect: '위험도 30% 감소',
        cost: '무료 (신한 지원)',
        duration: '1개월',
        priority: 'critical'
      },
      {
        title: '사업 구조 재편',
        description: '메뉴, 운영, 마케팅 전면 개편',
        expectedEffect: '수익성 20% 개선',
        cost: '200만원',
        duration: '2개월',
        priority: 'critical'
      },
      {
        title: '비용 절감 프로그램',
        description: '불필요한 지출 제거 및 효율성 극대화',
        expectedEffect: '고정비 15% 절감',
        cost: '50만원',
        duration: '1개월',
        priority: 'high'
      }
    ],
    financialProducts: [
      {
        name: '긴급 경영 안정화 대출',
        description: '폐업 위기 가맹점 특별 지원 프로그램',
        rate: '2.9%',
        limit: '최대 1,500만원',
        benefit: '12개월 거치, 최대 60개월 분할'
      },
      {
        name: '전문가 1:1 집중 컨설팅',
        description: '경영, 마케팅, 메뉴 전문가 패키지',
        rate: '무료',
        limit: '6개월 관리',
        benefit: '주 1회 방문 상담'
      },
      {
        name: '구조조정 지원 프로그램',
        description: '사업 재편 및 리브랜딩 종합 지원',
        rate: '3.2%',
        limit: '최대 2,000만원',
        benefit: '법무/세무 컨설팅 포함'
      }
    ]
  },

  '기타 위험': {
    description: '명확한 위험 유형으로 분류되지 않지만 주의가 필요한 상황입니다.',
    severity: 'low',
    icon: '⚠️',
    actions: [
      {
        title: '정기 경영 진단',
        description: '월 1회 경영 상태 체크 및 문제점 파악',
        expectedEffect: '위험 조기 발견',
        cost: '무료',
        duration: '지속',
        priority: 'medium'
      },
      {
        title: '예방적 마케팅',
        description: '고객 유지 및 매출 안정화 활동',
        expectedEffect: '안정적 성장 유지',
        cost: '20만원',
        duration: '지속',
        priority: 'low'
      }
    ],
    financialProducts: [
      {
        name: '안정 성장 지원 프로그램',
        description: '예방적 경영 관리 및 교육',
        rate: '무료',
        limit: '컨설팅 포함',
        benefit: '온라인 교육 제공'
      }
    ]
  }
}

// 위험 등급별 설명
export const riskLevelDescriptions = {
  'Very High': {
    description: '폐업 위험이 매우 높습니다. 즉각적인 조치가 필요합니다.',
    color: '#dc2626',
    urgency: '즉시 조치 필요',
    icon: '🚨'
  },
  'High': {
    description: '폐업 위험이 높습니다. 빠른 대응이 필요합니다.',
    color: '#ea580c',
    urgency: '조속한 조치 필요',
    icon: '⚠️'
  },
  'Medium': {
    description: '주의가 필요한 상태입니다. 개선 조치를 권장합니다.',
    color: '#ca8a04',
    urgency: '개선 권장',
    icon: '⚡'
  },
  'Low': {
    description: '현재는 양호한 상태입니다. 지속적인 관리가 필요합니다.',
    color: '#65a30d',
    urgency: '정기 관리',
    icon: '✓'
  },
  'Very Low': {
    description: '매우 안정적인 상태입니다. 현재 상태를 유지하세요.',
    color: '#16a34a',
    urgency: '안정',
    icon: '✓✓'
  }
}

// 우선순위별 설명
export const priorityDescriptions = {
  'critical': {
    label: '최우선 조치',
    description: '즉각적인 전문가 개입이 필요합니다',
    color: '#dc2626'
  },
  'important': {
    label: '중요',
    description: '빠른 시일 내 조치가 필요합니다',
    color: '#ea580c'
  },
  'watch': {
    label: '관찰 필요',
    description: '지속적인 모니터링이 필요합니다',
    color: '#ca8a04'
  },
  'normal': {
    label: '일반',
    description: '정기적인 관리가 필요합니다',
    color: '#65a30d'
  }
}
