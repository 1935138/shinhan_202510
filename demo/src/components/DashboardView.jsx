import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Papa from 'papaparse'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Label } from "@/components/ui/label"
import { ArrowLeft, TrendingUp, AlertTriangle, BarChart3, Store } from 'lucide-react'

function DashboardView() {
  const navigate = useNavigate()
  const [data, setData] = useState([])
  const [filteredData, setFilteredData] = useState([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({
    riskLevel: 'all',
    riskType: 'all',
    priority: 'all',
    area: 'all'
  })

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
            // 기본: 위험군(High, Very High) + 우선순위 높은 것만 표시
            const highRiskData = results.data.filter(item =>
              (item.risk_level === 'High' || item.risk_level === 'Very High') &&
              (item.priority === 'critical' || item.priority === 'important')
            )
            setFilteredData(highRiskData)
            setLoading(false)
          }
        })
      })
      .catch(error => {
        console.error('Error loading CSV:', error)
        setLoading(false)
      })
  }, [])

  // 필터링 적용
  useEffect(() => {
    let filtered = [...data]

    if (filters.riskLevel !== 'all') {
      filtered = filtered.filter(item => item.risk_level === filters.riskLevel)
    }
    if (filters.riskType !== 'all') {
      filtered = filtered.filter(item => item.risk_type === filters.riskType)
    }
    if (filters.priority !== 'all') {
      filtered = filtered.filter(item => item.priority === filters.priority)
    }
    if (filters.area !== 'all') {
      filtered = filtered.filter(item => item.HPSN_MCT_BZN_CD_NM === filters.area)
    }

    setFilteredData(filtered)
  }, [filters, data])

  // 통계 계산
  const stats = {
    total: filteredData.length,
    byRiskLevel: {},
    byRiskType: {},
    byPriority: {},
    avgRiskScore: 0
  }

  filteredData.forEach(item => {
    // 위험 등급별
    stats.byRiskLevel[item.risk_level] = (stats.byRiskLevel[item.risk_level] || 0) + 1
    // 위험 유형별
    stats.byRiskType[item.risk_type] = (stats.byRiskType[item.risk_type] || 0) + 1
    // 우선순위별
    stats.byPriority[item.priority] = (stats.byPriority[item.priority] || 0) + 1
    // 평균 위험 점수
    stats.avgRiskScore += parseFloat(item.risk_score)
  })

  if (filteredData.length > 0) {
    stats.avgRiskScore = (stats.avgRiskScore / filteredData.length).toFixed(1)
  }

  // 고유 값 추출 (필터 옵션용)
  const uniqueAreas = [...new Set(data.map(item => item.HPSN_MCT_BZN_CD_NM).filter(Boolean))].sort()
  const uniqueRiskLevels = [...new Set(data.map(item => item.risk_level))].sort()
  const uniqueRiskTypes = [...new Set(data.map(item => item.risk_type))].sort()
  const uniquePriorities = [...new Set(data.map(item => item.priority))].sort()

  // 차트 데이터 준비
  // 위험 등급 차트: Very Low 제외
  const riskLevelChartData = Object.entries(stats.byRiskLevel)
    .filter(([name]) => name !== 'Very Low')
    .map(([name, value]) => ({
      name,
      value,
      percentage: ((value / stats.total) * 100).toFixed(1)
    }))

  // 우선순위 차트: normal 제외
  const priorityChartData = Object.entries(stats.byPriority)
    .filter(([name]) => name !== 'normal')
    .map(([name, value]) => ({
      name,
      value,
      percentage: ((value / stats.total) * 100).toFixed(1)
    }))

  // 위험 유형 차트: "정상" 제외
  const riskTypeChartData = Object.entries(stats.byRiskType)
    .filter(([name]) => name !== '정상')
    .map(([name, value]) => ({
      name,
      value
    }))

  // 색상 팔레트 (Recharts용 hex, Badge용 Tailwind)
  const RISK_LEVEL_COLORS = {
    'Very High': '#dc2626',
    'High': '#ea580c',
    'Medium': '#ca8a04',
    'Low': '#65a30d',
    'Very Low': '#16a34a'
  }

  const PRIORITY_COLORS = {
    'critical': '#dc2626',
    'important': '#ea580c',
    'watch': '#ca8a04',
    'normal': '#65a30d'
  }

  // 신한 블루 단색 계열 (차분한 톤)
  const RISK_TYPE_COLORS = ['#0046FF', '#3D7CFF', '#6B9FFF', '#0037CC', '#5B7BA8', '#002999']

  // Badge용 Tailwind 클래스
  const getRiskLevelBadgeClass = (level) => {
    const colors = {
      'Very High': 'bg-red-600',
      'High': 'bg-orange-600',
      'Medium': 'bg-yellow-600',
      'Low': 'bg-green-600',
      'Very Low': 'bg-emerald-600'
    }
    return colors[level] || 'bg-gray-600'
  }

  const getPriorityBadgeClass = (priority) => {
    const colors = {
      'critical': 'bg-red-600',
      'important': 'bg-orange-600',
      'watch': 'bg-yellow-600',
      'normal': 'bg-green-600'
    }
    return colors[priority] || 'bg-gray-600'
  }

  // 커스텀 툴팁
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-semibold text-gray-900">{payload[0].name}</p>
          <p className="text-sm text-gray-600">
            {payload[0].value}개 ({payload[0].payload.percentage}%)
          </p>
        </div>
      )
    }
    return null
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-lg text-shinhan font-medium">데이터 로딩 중...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Navigation */}
        <div className="mb-8">
          <Button variant="outline" onClick={() => navigate('/')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            홈으로
          </Button>
        </div>

        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2 flex items-center justify-center gap-3">
            <Store className="h-10 w-10 text-shinhan" />
            가맹점 위험도 관리 대시보드
          </h1>
          <p className="text-lg text-gray-600">신한카드 빅콘테스트 2025 - 성동구 요식업 가맹점 위험도 분석</p>
        </div>

        {/* 통계 요약 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">위험군 가맹점</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-shinhan">{stats.total}</div>
              <p className="text-xs text-muted-foreground mt-1">우선 조치 대상</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">평균 위험 점수</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-orange-600">{stats.avgRiskScore}</div>
              <p className="text-xs text-muted-foreground mt-1">전체 평균</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">최우선 조치 필요</CardTitle>
              <AlertTriangle className="h-4 w-4 text-red-600" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-600">
                {stats.byPriority['critical'] || 0}
              </div>
              <p className="text-xs text-muted-foreground mt-1">critical 우선순위</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Very High 위험</CardTitle>
              <AlertTriangle className="h-4 w-4 text-orange-600" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-orange-600">
                {stats.byRiskLevel['Very High'] || 0}
              </div>
              <p className="text-xs text-muted-foreground mt-1">최고 위험도</p>
            </CardContent>
          </Card>
        </div>

        {/* 차트 그리드 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* 위험 등급별 파이 차트 */}
          <Card>
            <CardHeader>
              <CardTitle>위험 등급 분포</CardTitle>
              <CardDescription>가맹점의 위험도 등급별 분포</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskLevelChartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name} (${percentage}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskLevelChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={RISK_LEVEL_COLORS[entry.name]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              {stats.byRiskLevel['Very Low'] && (
                <p className="text-xs text-muted-foreground mt-2">
                  제외: Very Low ({stats.byRiskLevel['Very Low']}개)
                </p>
              )}
            </CardContent>
          </Card>

          {/* 우선순위별 도넛 차트 */}
          <Card>
            <CardHeader>
              <CardTitle>우선순위 분포</CardTitle>
              <CardDescription>조치 우선순위별 가맹점 분포</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={priorityChartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name} (${percentage}%)`}
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {priorityChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={PRIORITY_COLORS[entry.name]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              {stats.byPriority['normal'] && (
                <p className="text-xs text-muted-foreground mt-2">
                  제외: normal ({stats.byPriority['normal']}개)
                </p>
              )}
            </CardContent>
          </Card>

          {/* 위험 유형별 바 차트 */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>위험 유형별 가맹점 수</CardTitle>
              <CardDescription>각 위험 유형에 해당하는 가맹점 수</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={riskTypeChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#0046FF">
                    {riskTypeChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={RISK_TYPE_COLORS[index % RISK_TYPE_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              {stats.byRiskType['정상'] && (
                <p className="text-xs text-muted-foreground mt-2">
                  제외: 정상 ({stats.byRiskType['정상']}개)
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* 필터 */}
        <Card className="mb-8">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>필터</CardTitle>
                <CardDescription>조건을 선택하여 데이터를 필터링하세요</CardDescription>
              </div>
              <Badge variant="outline" className="text-sm">
                📊 기본: High/Very High + critical/important만 표시
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="risk-level">위험 등급</Label>
                <Select
                  value={filters.riskLevel}
                  onValueChange={(value) => setFilters({...filters, riskLevel: value})}
                >
                  <SelectTrigger id="risk-level">
                    <SelectValue placeholder="전체" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체</SelectItem>
                    {uniqueRiskLevels.map(level => (
                      <SelectItem key={level} value={level}>{level}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="risk-type">위험 유형</Label>
                <Select
                  value={filters.riskType}
                  onValueChange={(value) => setFilters({...filters, riskType: value})}
                >
                  <SelectTrigger id="risk-type">
                    <SelectValue placeholder="전체" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체</SelectItem>
                    {uniqueRiskTypes.map(type => (
                      <SelectItem key={type} value={type}>{type}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="priority">우선순위</Label>
                <Select
                  value={filters.priority}
                  onValueChange={(value) => setFilters({...filters, priority: value})}
                >
                  <SelectTrigger id="priority">
                    <SelectValue placeholder="전체" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체</SelectItem>
                    {uniquePriorities.map(priority => (
                      <SelectItem key={priority} value={priority}>{priority}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="area">상권</Label>
                <Select
                  value={filters.area}
                  onValueChange={(value) => setFilters({...filters, area: value})}
                >
                  <SelectTrigger id="area">
                    <SelectValue placeholder="전체" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체</SelectItem>
                    {uniqueAreas.map(area => (
                      <SelectItem key={area} value={area}>{area}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 데이터 테이블 */}
        <Card>
          <CardHeader>
            <CardTitle>가맹점 목록 ({filteredData.length}개)</CardTitle>
            <CardDescription>필터링된 가맹점의 상세 정보</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>가맹점 ID</TableHead>
                    <TableHead>상권</TableHead>
                    <TableHead className="text-right">위험 점수</TableHead>
                    <TableHead>위험 등급</TableHead>
                    <TableHead>위험 유형</TableHead>
                    <TableHead>우선순위</TableHead>
                    <TableHead className="text-right">신뢰도</TableHead>
                    <TableHead className="text-right">폐업 확률</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredData.slice(0, 100).map((item, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-mono text-xs">{item.ENCODED_MCT}</TableCell>
                      <TableCell>{item.HPSN_MCT_BZN_CD_NM || '-'}</TableCell>
                      <TableCell className="text-right font-medium">{item.risk_score}</TableCell>
                      <TableCell>
                        <Badge className={getRiskLevelBadgeClass(item.risk_level)}>
                          {item.risk_level}
                        </Badge>
                      </TableCell>
                      <TableCell className="max-w-xs truncate">{item.risk_type}</TableCell>
                      <TableCell>
                        <Badge className={getPriorityBadgeClass(item.priority)}>
                          {item.priority}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {(parseFloat(item.classification_confidence) * 100).toFixed(0)}%
                      </TableCell>
                      <TableCell className="text-right font-medium text-red-600">
                        {(parseFloat(item.closure_probability) * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {filteredData.length > 100 && (
              <p className="text-sm text-muted-foreground mt-4">
                * 상위 100개 가맹점만 표시됩니다
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default DashboardView
