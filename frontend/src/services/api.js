/**
 * API Service for Merchant Risk Prediction System
 *
 * This service handles all communication with the Flask backend API
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

/**
 * Helper function to handle API responses
 */
async function handleResponse(response) {
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'API request failed');
  }
  return response.json();
}

/**
 * Health check endpoint
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(response);
}

/**
 * Get list of all merchants (for testing)
 */
export async function listMerchants() {
  const response = await fetch(`${API_BASE_URL}/merchants`);
  return handleResponse(response);
}

/**
 * Get risk prediction for a specific merchant
 *
 * @param {string} merchantId - Merchant ID (ENCODED_MCT)
 * @param {string|null} date - Optional date in YYYYMM format
 * @returns {Promise<Object>} Risk prediction data
 */
export async function getMerchantRisk(merchantId, date = null) {
  const url = new URL(`${API_BASE_URL}/merchant/${merchantId}/risk`);
  if (date) {
    url.searchParams.append('date', date);
  }

  const response = await fetch(url);
  return handleResponse(response);
}

/**
 * Get top risk signals for a merchant
 *
 * @param {string} merchantId - Merchant ID
 * @returns {Promise<Object>} Risk signals data
 */
export async function getMerchantSignals(merchantId) {
  const response = await fetch(`${API_BASE_URL}/merchant/${merchantId}/signals`);
  return handleResponse(response);
}

/**
 * Get sales history for a merchant
 *
 * @param {string} merchantId - Merchant ID
 * @returns {Promise<Object>} Sales history data
 */
export async function getMerchantSales(merchantId) {
  const response = await fetch(`${API_BASE_URL}/merchant/${merchantId}/sales`);
  return handleResponse(response);
}

/**
 * Get region overview statistics
 *
 * @returns {Promise<Object>} Region overview data
 */
export async function getRegionOverview() {
  const response = await fetch(`${API_BASE_URL}/regions/overview`);
  return handleResponse(response);
}

/**
 * Transform API risk data to HomeScreen format
 */
export function transformRiskDataForHome(riskData, signalsData) {
  return {
    healthData: {
      score: riskData.risk_score,
      status: riskData.status,
      daysUntilCrisis: riskData.days_until_crisis,
    },
    riskSignals: signalsData.signals.map(signal => ({
      icon: signal.icon,
      title: signal.title,
      description: signal.description,
      color: signal.color,
    })),
  };
}

/**
 * Transform API sales data to DetailReportScreen format
 */
export function transformSalesDataForDetail(salesData, riskData) {
  return {
    salesData: salesData.sales_data,
    customerData: {
      returning: 45,  // TODO: Get from API when available
      new: 55,
      ageDistribution: [
        { age: '20대', percentage: 15 },
        { age: '30대', percentage: 35 },
        { age: '40대', percentage: 30 },
        { age: '50대', percentage: 15 },
        { age: '60대+', percentage: 5 },
      ],
    },
    comparisonData: {
      industryAverage: 85,
      ranking: 15,
      totalStores: 50,
      additionalMetrics: [
        { label: '전월 대비', value: '-5%' },
        { label: '전년 대비', value: '+12%' },
      ],
    },
  };
}

/**
 * Transform API data for WebDashboardScreen
 */
export function transformDataForDashboard(riskData, salesData, signalsData) {
  return {
    healthData: {
      score: riskData.risk_score,
      status: riskData.status,
      daysUntilCrisis: riskData.days_until_crisis,
    },
    salesData: salesData.sales_data,
    customerData: {
      returning: 45,
      new: 55,
    },
    shapData: signalsData.signals.map((signal, idx) => ({
      label: signal.title,
      value: 0.25 - (idx * 0.05),  // Mock SHAP values
    })),
  };
}

/**
 * Transform API data for AdminConsoleScreen
 */
export function transformDataForAdmin(overviewData) {
  return {
    regionName: overviewData.region_name,
    totalStores: overviewData.total_stores,
    statusBreakdown: overviewData.status_breakdown,
    riskMerchants: overviewData.risk_merchants.map(merchant => ({
      name: `가맹점 ${merchant.merchant_id.slice(-4)}`,  // Mask merchant ID
      merchant_id: merchant.merchant_id,  // Keep for API calls
      category: merchant.category,
      riskScore: merchant.risk_score,
      riskType: merchant.risk_type,
      lastConsultDate: null,  // TODO: Add to backend
      isUrgent: merchant.is_urgent,
    })),
  };
}

export default {
  checkHealth,
  listMerchants,
  getMerchantRisk,
  getMerchantSignals,
  getMerchantSales,
  getRegionOverview,
  transformRiskDataForHome,
  transformSalesDataForDetail,
  transformDataForDashboard,
  transformDataForAdmin,
};
