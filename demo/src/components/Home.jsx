import { useNavigate } from 'react-router-dom'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Store, Building2, TrendingUp, Shield, BarChart3, Filter } from 'lucide-react'

function Home() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-gradient-to-br from-shinhan via-shinhan-600 to-shinhan-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="text-center mb-16 pt-8">
          <h1 className="text-5xl font-bold text-white mb-4 drop-shadow-lg">
            가맹점 위기 조기 경보 시스템
          </h1>
          <p className="text-xl text-white/90">
            신한카드 빅콘테스트 2025 - AI 기반 가맹점 위험도 분석
          </p>
        </header>

        {/* Cards Grid */}
        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Merchant Card */}
          <Card
            className="hover:shadow-2xl transition-all duration-300 cursor-pointer border-2 hover:border-shinhan hover:scale-105"
            onClick={() => navigate('/merchant')}
          >
            <CardHeader className="pb-4">
              <div className="w-16 h-16 bg-shinhan-50 rounded-full flex items-center justify-center mb-4 mx-auto">
                <Store className="w-8 h-8 text-shinhan" />
              </div>
              <CardTitle className="text-center text-2xl">가맹점 주인</CardTitle>
              <CardDescription className="text-center text-base pt-2">
                내 가맹점의 위험도를 확인하고<br/>
                맞춤형 개선 방안을 받아보세요
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 pb-6">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <TrendingUp className="w-4 h-4 text-shinhan" />
                <span>건강 점수 확인</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <Shield className="w-4 h-4 text-shinhan" />
                <span>위험 신호 파악</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <BarChart3 className="w-4 h-4 text-shinhan" />
                <span>맞춤 솔루션 제공</span>
              </div>
              <Button className="w-full mt-4" size="lg">
                내 가맹점 조회하기 →
              </Button>
            </CardContent>
          </Card>

          {/* Dashboard Card */}
          <Card
            className="hover:shadow-2xl transition-all duration-300 cursor-pointer border-2 hover:border-shinhan hover:scale-105"
            onClick={() => navigate('/dashboard')}
          >
            <CardHeader className="pb-4">
              <div className="w-16 h-16 bg-purple-50 rounded-full flex items-center justify-center mb-4 mx-auto">
                <Building2 className="w-8 h-8 text-purple-600" />
              </div>
              <CardTitle className="text-center text-2xl">카드사 담당자</CardTitle>
              <CardDescription className="text-center text-base pt-2">
                전체 가맹점 현황을 모니터링하고<br/>
                위험군을 관리하세요
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 pb-6">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <BarChart3 className="w-4 h-4 text-purple-600" />
                <span>전체 현황 대시보드</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <Filter className="w-4 h-4 text-purple-600" />
                <span>위험군 필터링</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <TrendingUp className="w-4 h-4 text-purple-600" />
                <span>상세 분석 리포트</span>
              </div>
              <Button className="w-full mt-4 bg-purple-600 hover:bg-purple-700" size="lg">
                관리자 대시보드 열기 →
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 pb-8">
          <p className="text-white/80 text-sm">데모 버전 • 실제 서비스와 다를 수 있습니다</p>
        </footer>
      </div>
    </div>
  )
}

export default Home
