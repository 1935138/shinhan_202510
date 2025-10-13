import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Papa from 'papaparse'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, TrendingUp, DollarSign, Calendar, Target, ArrowLeft, Home, Search } from 'lucide-react'
import { riskTypeSolutions, riskLevelDescriptions } from '../data/riskTypeSolutions'

function MerchantView() {
  const navigate = useNavigate()
  const { merchantId } = useParams()
  const [searchInput, setSearchInput] = useState(merchantId || '')
  const [data, setData] = useState([])
  const [merchantData, setMerchantData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)

  // CSV 데이터 로드
  useEffect(() => {
    fetch('/risk_classification_results.csv')
      .then(response => response.text())
      .then(csvText => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setData(results.data)
            setLoading(false)

            if (merchantId) {
              searchMerchant(merchantId, results.data)
            }
          }
        })
      })
      .catch(error => {
        console.error('Error loading CSV:', error)
        setLoading(false)
      })
  }, [merchantId])

  const searchMerchant = (id, dataToSearch = data) => {
    if (!id.trim()) return

    const found = dataToSearch.find(
      item => item.ENCODED_MCT.toLowerCase() === id.trim().toLowerCase()
    )

    if (found) {
      setMerchantData(found)
      setNotFound(false)
      navigate(`/merchant/${found.ENCODED_MCT}`)
    } else {
      setMerchantData(null)
      setNotFound(true)
    }
  }

  const handleSearch = (e) => {
    e.preventDefault()
    searchMerchant(searchInput)
  }

  const handleReset = () => {
    setSearchInput('')
    setMerchantData(null)
    setNotFound(false)
    navigate('/merchant')
  }

  const getRiskLevelColor = (level) => {
    const colors = {
      'Very High': 'bg-red-600',
      'High': 'bg-orange-600',
      'Medium': 'bg-yellow-600',
      'Low': 'bg-green-600',
      'Very Low': 'bg-emerald-600'
    }
    return colors[level] || 'bg-gray-600'
  }

  const getPriorityColor = (priority) => {
    const colors = {
      'critical': 'bg-red-600',
      'important': 'bg-orange-600',
      'watch': 'bg-yellow-600',
      'normal': 'bg-green-600'
    }
    return colors[priority] || 'bg-gray-600'
  }

  // 로딩 화면
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-shinhan-100 to-shinhan-50 flex items-center justify-center">
        <div className="text-lg text-shinhan font-medium">데이터 로딩 중...</div>
      </div>
    )
  }

  // 검색 화면
  if (!merchantData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-shinhan via-shinhan-600 to-shinhan-800">
        <div className="container mx-auto px-4 py-8 max-w-4xl">
          <Button variant="ghost" onClick={() => navigate('/')} className="text-white hover:text-white hover:bg-white/20 mb-8">
            <ArrowLeft className="mr-2 h-4 w-4" />
            홈으로
          </Button>

          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">🏪 내 가맹점 조회</h1>
            <p className="text-xl text-white/90">가맹점 번호를 입력하여 위험도를 확인하세요</p>
          </div>

          <Card className="mb-8">
            <CardContent className="pt-6">
              <form onSubmit={handleSearch} className="flex gap-2">
                <Input
                  type="text"
                  placeholder="가맹점 번호 (예: 1A9644F28E)"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="text-lg"
                  autoFocus
                />
                <Button type="submit" size="lg" className="px-8">
                  <Search className="mr-2 h-4 w-4" />
                  조회하기
                </Button>
              </form>

              {notFound && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-red-800 font-medium flex items-center gap-2">
                    <AlertCircle className="h-4 w-4" />
                    해당 가맹점 번호를 찾을 수 없습니다.
                  </p>
                  <p className="text-sm text-red-600 mt-1">가맹점 번호를 다시 확인해주세요.</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>예시 가맹점 번호</CardTitle>
              <CardDescription>아래 번호를 클릭하여 바로 조회할 수 있습니다</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {data.slice(0, 6).map((item, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    className="justify-between h-auto py-3"
                    onClick={() => {
                      setSearchInput(item.ENCODED_MCT)
                      searchMerchant(item.ENCODED_MCT)
                    }}
                  >
                    <span className="font-mono text-sm">{item.ENCODED_MCT}</span>
                    <Badge className={getRiskLevelColor(item.risk_level)}>
                      {item.risk_level}
                    </Badge>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  // 가맹점 상세 화면
  const solution = riskTypeSolutions[merchantData.risk_type] || riskTypeSolutions['기타 위험']
  const riskLevelInfo = riskLevelDescriptions[merchantData.risk_level]
  const closureProbability = (parseFloat(merchantData.closure_probability) * 100).toFixed(1)
  const confidence = (parseFloat(merchantData.classification_confidence) * 100).toFixed(0)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* 헤더 */}
        <div className="flex justify-between items-center mb-8">
          <Button variant="outline" onClick={handleReset}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            다른 가맹점 조회
          </Button>
          <Button variant="ghost" onClick={() => navigate('/')}>
            <Home className="mr-2 h-4 w-4" />
            홈으로
          </Button>
        </div>

        {/* 건강 점수 카드 */}
        <Card className="mb-6 border-2 border-shinhan/20">
          <CardContent className="pt-6">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="text-center flex flex-col items-center justify-center border-r border-gray-200">
                <p className="text-lg text-muted-foreground mb-2">내 가맹점 건강 점수</p>
                <div className={`text-7xl font-bold mb-4 ${merchantData.risk_score >= 60 ? 'text-red-600' : 'text-orange-600'}`}>
                  {merchantData.risk_score}점
                </div>
                <Badge className={`${getRiskLevelColor(merchantData.risk_level)} text-lg px-4 py-2`}>
                  {riskLevelInfo?.icon} {merchantData.risk_level}
                </Badge>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">가맹점 번호</span>
                  <span className="font-mono font-medium">{merchantData.ENCODED_MCT}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">상권</span>
                  <span className="font-medium">{merchantData.HPSN_MCT_BZN_CD_NM || '미분류'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">폐업 확률</span>
                  <span className="font-bold text-red-600">{closureProbability}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">분석 신뢰도</span>
                  <span className="font-medium">{confidence}%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 위험 수준 설명 */}
        <Card className="mb-6 bg-yellow-50 border-yellow-200">
          <CardContent className="pt-6 text-center">
            <h2 className="text-2xl font-bold text-yellow-900 mb-2">{riskLevelInfo?.urgency}</h2>
            <p className="text-yellow-800">{riskLevelInfo?.description}</p>
          </CardContent>
        </Card>

        {/* 위험 유형 카드 */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center gap-4">
              <div className="text-5xl">{solution.icon}</div>
              <div className="flex-1">
                <CardTitle className="text-2xl mb-2">위험 유형: {merchantData.risk_type}</CardTitle>
                <CardDescription className="text-base">{solution.description}</CardDescription>
              </div>
              <Badge className={`${getPriorityColor(merchantData.priority)} text-base px-4 py-2`}>
                우선순위: {merchantData.priority}
              </Badge>
            </div>
          </CardHeader>
        </Card>

        {/* 맞춤 솔루션 */}
        <div className="mb-6">
          <h2 className="text-3xl font-bold mb-6 text-center">💡 맞춤 개선 방안</h2>
          <div className="grid md:grid-cols-3 gap-4">
            {solution.actions.map((action, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-lg">{index + 1}. {action.title}</CardTitle>
                    {action.priority === 'critical' && (
                      <Badge variant="destructive" className="text-xs">긴급</Badge>
                    )}
                  </div>
                  <CardDescription>{action.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-start gap-2 text-sm">
                    <TrendingUp className="h-4 w-4 text-shinhan mt-0.5" />
                    <div>
                      <strong>예상 효과:</strong> {action.expectedEffect}
                    </div>
                  </div>
                  <div className="flex items-start gap-2 text-sm">
                    <DollarSign className="h-4 w-4 text-shinhan mt-0.5" />
                    <div>
                      <strong>소요 비용:</strong> {action.cost}
                    </div>
                  </div>
                  <div className="flex items-start gap-2 text-sm">
                    <Calendar className="h-4 w-4 text-shinhan mt-0.5" />
                    <div>
                      <strong>실행 기간:</strong> {action.duration}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* 추천 금융상품 */}
        {solution.financialProducts && solution.financialProducts.length > 0 && (
          <div className="mb-6">
            <h2 className="text-3xl font-bold mb-6 text-center">💰 추천 금융상품</h2>
            <div className="grid md:grid-cols-2 gap-4">
              {solution.financialProducts.map((product, index) => (
                <Card key={index} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <CardTitle>{product.name}</CardTitle>
                    <CardDescription>{product.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="bg-muted p-3 rounded-lg space-y-2 text-sm">
                      <div><strong>금리:</strong> {product.rate}</div>
                      <div><strong>한도:</strong> {product.limit}</div>
                      <div className="text-shinhan font-medium"><strong>혜택:</strong> {product.benefit}</div>
                    </div>
                    <Button className="w-full">상담 신청하기</Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* CTA */}
        <Card className="bg-gradient-to-r from-shinhan to-shinhan-600 text-white">
          <CardContent className="pt-6 text-center">
            <h2 className="text-3xl font-bold mb-2">전문가 상담이 필요하신가요?</h2>
            <p className="text-xl mb-6 text-white/90">위기 극복을 위한 1:1 맞춤 컨설팅을 제공합니다</p>
            <Button size="lg" variant="secondary" className="px-8 py-6 text-lg">
              <Target className="mr-2 h-5 w-5" />
              무료 상담 신청하기
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default MerchantView
