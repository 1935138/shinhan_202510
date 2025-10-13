import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './components/Home'
import MerchantView from './components/MerchantView'
import DashboardView from './components/DashboardView'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/merchant" element={<MerchantView />} />
        <Route path="/merchant/:merchantId" element={<MerchantView />} />
        <Route path="/dashboard" element={<DashboardView />} />
      </Routes>
    </Router>
  )
}

export default App
