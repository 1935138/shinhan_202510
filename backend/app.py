"""
Flask Backend API for Merchant Risk Prediction System

This API serves the All Interval Features XGBoost model for predicting
merchant closure risk and provides merchant-specific risk analysis.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'

# Global variables for model and data
model = None
featured_data = None
feature_cols = None
merchant_metadata = None


def load_model_and_data():
    """Load XGBoost model and feature data on startup"""
    global model, featured_data, feature_cols, merchant_metadata

    print("Loading model and data...")

    # Load featured data with all interval features
    data_path = DATA_DIR / 'featured_data_with_intervals.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    featured_data = pd.read_csv(data_path)
    print(f"✅ Loaded data: {featured_data.shape}")

    # Define feature columns (exclude metadata and targets)
    exclude_cols = [
        'ENCODED_MCT', 'TA_YM', 'MCT_ME_D', 'MCT_BSE_AR', 'MCT_NM',
        'MCT_BRD_NUM', 'MCT_SIGUNGU_NM', 'HPSN_MCT_ZCD_NM', 'HPSN_MCT_BZN_CD_NM', 'ARE_D',
        'will_close_1m', 'will_close_3m', 'months_until_close', 'is_valid_for_training'
    ]
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    print(f"✅ Feature columns: {len(feature_cols)}")

    # Extract merchant metadata
    merchant_metadata = featured_data[['ENCODED_MCT', 'TA_YM', 'HPSN_MCT_BZN_CD_NM']].copy()
    merchant_metadata = merchant_metadata.rename(columns={'HPSN_MCT_BZN_CD_NM': 'category'})

    # Load trained model (we'll train it if not exists)
    model_path = MODEL_DIR / 'xgboost_all_interval_model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Loaded model from: {model_path}")
    else:
        print("⚠️  Model not found. Training new model...")
        train_model()

    print("✅ Model and data loaded successfully!")


def train_model():
    """Train XGBoost model with All Interval Features"""
    global model

    print("\n" + "="*80)
    print("TRAINING ALL INTERVAL FEATURES MODEL")
    print("="*80)

    # Filter valid training data
    df_train = featured_data[featured_data['is_valid_for_training'] == 1].copy()

    # Split by time
    train_mask = df_train['TA_YM'] <= 202406
    valid_mask = (df_train['TA_YM'] > 202406) & (df_train['TA_YM'] <= 202409)

    X_train = df_train[train_mask][feature_cols]
    y_train = df_train[train_mask]['will_close_3m']

    X_valid = df_train[valid_mask][feature_cols]
    y_valid = df_train[valid_mask]['will_close_3m']

    print(f"Train: {len(X_train)} rows, Positive: {y_train.sum()}")
    print(f"Valid: {len(X_valid)} rows, Positive: {y_valid.sum()}")

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale positive weight: {scale_pos_weight:.1f}")

    # Train XGBoost
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist'
    )

    print("\nTraining model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # Save model
    model_path = MODEL_DIR / 'xgboost_all_interval_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")
    print("="*80)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': featured_data is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/merchant/<merchant_id>/risk', methods=['GET'])
def get_merchant_risk(merchant_id):
    """
    Get risk prediction for a specific merchant

    Query params:
    - date: YYYYMM format (optional, defaults to latest)
    """
    try:
        # Get date parameter
        date_param = request.args.get('date', None)

        # Filter merchant data
        merchant_data = featured_data[featured_data['ENCODED_MCT'] == merchant_id].copy()

        if len(merchant_data) == 0:
            return jsonify({'error': f'Merchant {merchant_id} not found'}), 404

        # Get latest or specific date
        if date_param:
            merchant_data = merchant_data[merchant_data['TA_YM'] == int(date_param)]
            if len(merchant_data) == 0:
                return jsonify({'error': f'No data for merchant {merchant_id} on {date_param}'}), 404
        else:
            merchant_data = merchant_data.sort_values('TA_YM').tail(1)

        # Get features
        X = merchant_data[feature_cols]

        # Predict
        risk_proba = model.predict_proba(X)[:, 1][0]
        risk_score = int(risk_proba * 100)

        # Determine status and days until crisis
        if risk_score >= 80:
            status = 'danger'
            days_until_crisis = 30
        elif risk_score >= 60:
            status = 'warning'
            days_until_crisis = 45
        elif risk_score >= 40:
            status = 'caution'
            days_until_crisis = 60
        else:
            status = 'safe'
            days_until_crisis = None

        # Get merchant info
        merchant_info = merchant_data.iloc[0]
        category = merchant_info.get('HPSN_MCT_BZN_CD_NM', 'Unknown')
        ta_ym = int(merchant_info['TA_YM'])

        return jsonify({
            'merchant_id': merchant_id,
            'category': category,
            'date': ta_ym,
            'risk_score': risk_score,
            'risk_probability': float(risk_proba),
            'status': status,
            'days_until_crisis': days_until_crisis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merchant/<merchant_id>/signals', methods=['GET'])
def get_risk_signals(merchant_id):
    """
    Get top risk signals for a merchant based on interval pattern features
    """
    try:
        # Get merchant data
        merchant_data = featured_data[featured_data['ENCODED_MCT'] == merchant_id].copy()

        if len(merchant_data) == 0:
            return jsonify({'error': f'Merchant {merchant_id} not found'}), 404

        # Get latest data
        merchant_data = merchant_data.sort_values('TA_YM').tail(1)

        # Analyze interval pattern features to identify risk signals
        signals = []
        latest = merchant_data.iloc[0]

        # Signal 1: Consecutive declines
        decline_features = [col for col in feature_cols if 'consecutive_decline' in col]
        max_consecutive_decline = 0
        decline_metric = None

        for col in decline_features:
            val = latest[col]
            if val > max_consecutive_decline:
                max_consecutive_decline = val
                decline_metric = col.split('_consecutive')[0]

        if max_consecutive_decline >= 3:
            signals.append({
                'icon': 'trending-down',
                'title': f'{int(max_consecutive_decline)}개월 연속 성과 하락',
                'description': f'{decline_metric} 지표가 지속적으로 악화',
                'color': 'red'
            })

        # Signal 2: At worst performance
        worst_features = [col for col in feature_cols if 'at_worst_now' in col]
        at_worst_count = sum([latest[col] for col in worst_features if col in latest.index])

        if at_worst_count >= 2:
            signals.append({
                'icon': 'alert-triangle',
                'title': '역대 최악 성과 기록 중',
                'description': f'{int(at_worst_count)}개 지표가 역대 최저',
                'color': 'red'
            })

        # Signal 3: Distance from best
        distance_features = [col for col in feature_cols if 'distance_from_best' in col]
        max_distance = 0
        distance_metric = None

        for col in distance_features:
            if col in latest.index:
                val = latest[col]
                if val > max_distance:
                    max_distance = val
                    distance_metric = col.split('_distance')[0]

        if max_distance >= 3:
            signals.append({
                'icon': 'trending-down',
                'title': f'전성기 대비 {int(max_distance)}단계 하락',
                'description': f'{distance_metric} 최고 성과 대비 큰 폭 하락',
                'color': 'orange'
            })

        # Signal 4: Aligned decline (cross-metric)
        aligned_features = [col for col in feature_cols if 'aligned_decline' in col]
        aligned_decline_count = sum([latest[col] for col in aligned_features if col in latest.index])

        if aligned_decline_count >= 1:
            signals.append({
                'icon': 'alert-circle',
                'title': '복합 지표 동반 하락',
                'description': '매출, 고객수 등 여러 지표가 동시 악화',
                'color': 'red'
            })

        # Signal 5: High volatility
        volatility_features = [col for col in feature_cols if 'volatility' in col]
        high_volatility_count = 0

        for col in volatility_features:
            if col in latest.index:
                val = latest[col]
                if val > 1.5:  # High volatility threshold
                    high_volatility_count += 1

        if high_volatility_count >= 2:
            signals.append({
                'icon': 'activity',
                'title': '불안정한 성과 변동',
                'description': '성과 지표가 급격히 변동하며 불안정',
                'color': 'amber'
            })

        # If no signals detected, add generic ones
        if len(signals) == 0:
            signals = [
                {
                    'icon': 'info',
                    'title': '지표 모니터링 필요',
                    'description': '현재 명확한 위험 신호는 감지되지 않음',
                    'color': 'blue'
                }
            ]

        return jsonify({
            'merchant_id': merchant_id,
            'signals': signals[:3],  # Top 3 signals
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merchant/<merchant_id>/sales', methods=['GET'])
def get_sales_history(merchant_id):
    """Get sales history for a merchant (last 12 months)"""
    try:
        # Get merchant data
        merchant_data = featured_data[featured_data['ENCODED_MCT'] == merchant_id].copy()

        if len(merchant_data) == 0:
            return jsonify({'error': f'Merchant {merchant_id} not found'}), 404

        # Get last 12 months
        merchant_data = merchant_data.sort_values('TA_YM').tail(12)

        # Extract sales data (using RC_M1_SAA as proxy for sales performance)
        sales_data = []
        for _, row in merchant_data.iterrows():
            month = str(row['TA_YM'])[-2:]  # Get month
            month_name = f"{month}월"

            # Use interval as inverse sales score (lower interval = higher sales)
            # Convert interval (1-6) to relative sales score
            interval = row.get('RC_M1_SAA', 3)
            sales_score = (7 - interval) * 1000  # Higher = better

            sales_data.append({
                'month': month_name,
                'sales': int(sales_score)
            })

        return jsonify({
            'merchant_id': merchant_id,
            'sales_data': sales_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/regions/overview', methods=['GET'])
def get_region_overview():
    """Get overview statistics for all merchants in region"""
    try:
        # Get latest data for all merchants
        latest_data = featured_data.sort_values(['ENCODED_MCT', 'TA_YM']).groupby('ENCODED_MCT').tail(1)

        # Predict risk for all merchants
        X = latest_data[feature_cols]
        risk_probas = model.predict_proba(X)[:, 1]
        latest_data['risk_score'] = (risk_probas * 100).astype(int)

        # Categorize by risk status
        def get_status(score):
            if score >= 80:
                return 'danger'
            elif score >= 60:
                return 'warning'
            elif score >= 40:
                return 'caution'
            elif score >= 20:
                return 'good'
            else:
                return 'safe'

        latest_data['status'] = latest_data['risk_score'].apply(get_status)

        # Count by status
        status_counts = latest_data['status'].value_counts()
        total_stores = len(latest_data)

        status_breakdown = []
        status_map = {
            'safe': {'label': '안전', 'color': 'safe'},
            'good': {'label': '양호', 'color': 'good'},
            'caution': {'label': '주의', 'color': 'caution'},
            'warning': {'label': '경고', 'color': 'warning'},
            'danger': {'label': '위험', 'color': 'danger'}
        }

        for status_key, status_info in status_map.items():
            count = status_counts.get(status_key, 0)
            percentage = (count / total_stores * 100) if total_stores > 0 else 0
            status_breakdown.append({
                'status': status_info['label'],
                'count': int(count),
                'percentage': round(percentage, 1),
                'color': status_info['color']
            })

        # Get high-risk merchants
        high_risk = latest_data[latest_data['risk_score'] >= 70].sort_values('risk_score', ascending=False).head(10)

        risk_merchants = []
        for _, row in high_risk.iterrows():
            risk_merchants.append({
                'merchant_id': row['ENCODED_MCT'],
                'category': row.get('HPSN_MCT_BZN_CD_NM', 'Unknown'),
                'risk_score': int(row['risk_score']),
                'risk_type': '매출급락형',  # TODO: Determine from features
                'is_urgent': row['risk_score'] >= 80
            })

        return jsonify({
            'region_name': '성동구',
            'total_stores': int(total_stores),
            'status_breakdown': status_breakdown,
            'risk_merchants': risk_merchants,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merchants', methods=['GET'])
def list_merchants():
    """List all merchant IDs (for testing)"""
    try:
        merchant_ids = featured_data['ENCODED_MCT'].unique().tolist()
        return jsonify({
            'total': len(merchant_ids),
            'merchants': merchant_ids[:100],  # Return first 100
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model and data on startup
    load_model_and_data()

    # Run Flask app
    print("\n" + "="*80)
    print("🚀 Starting Flask API Server")
    print("="*80)
    print("Endpoints available:")
    print("  - GET /api/health")
    print("  - GET /api/merchants")
    print("  - GET /api/merchant/<id>/risk")
    print("  - GET /api/merchant/<id>/signals")
    print("  - GET /api/merchant/<id>/sales")
    print("  - GET /api/regions/overview")
    print("="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
