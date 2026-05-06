"""
UPI Fraud Detection API
Sequential LSTM-CNN with Attention is the primary detector.
Business rules are minimal nudges only.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import keras
from keras import layers
import numpy as np
import pandas as pd
import joblib
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from database import init_db, log_fraud_case, get_all_fraud_cases, get_fraud_stats
from alerts import send_fraud_alert, should_alert

# ── App setup ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
init_db()

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'model')

# ── Load meta ──────────────────────────────────────────────
meta            = joblib.load(os.path.join(MODEL_DIR, 'best_model_meta.pkl'))
MODEL_NAME      = meta['name']
MODEL_TYPE      = meta['type']
FRAUD_THRESHOLD = meta['threshold']

class AttentionLayer(layers.Layer):
    """Same class used during training — required for model loading."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True, name='attention_W'
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True, name='attention_b'
        )
        self.u = self.add_weight(
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True, name='attention_u'
        )
        super().build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.matmul(x, self.W) + self.b)
        ait = tf.matmul(uit, tf.expand_dims(self.u, -1))
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait)
        ait = tf.expand_dims(ait, -1)
        weighted = x * ait
        return tf.reduce_sum(weighted, axis=1)

    def get_config(self):
        return super().get_config()
    
# ── Load model ─────────────────────────────────────────────
if MODEL_TYPE == 'keras':
    model = keras.models.load_model(
        os.path.join(MODEL_DIR, 'best_model.h5'),
        custom_objects={'AttentionLayer': AttentionLayer},
        compile=False
    )
elif MODEL_TYPE == 'kmeans':
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
else:
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))

label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders_v2.pkl'))
scaler         = joblib.load(os.path.join(MODEL_DIR, 'scaler_v2.pkl'))

print(f"Loaded: {MODEL_NAME} | type={MODEL_TYPE} | threshold={FRAUD_THRESHOLD}")

# ── Feature config — must match notebook exactly ───────────
CATEGORICAL_COLS = [
    'transaction type', 'merchant_category', 'sender_age_group',
    'receiver_age_group', 'sender_state', 'sender_bank',
    'receiver_bank', 'device_type', 'network_type', 'day_of_week'
]
SCALE_COLS   = ['amount (INR)', 'hour_of_day']
FEATURE_COLS = [
    'transaction type', 'merchant_category', 'amount (INR)',
    'sender_age_group', 'receiver_age_group', 'sender_state',
    'sender_bank', 'receiver_bank', 'device_type', 'network_type',
    'hour_of_day', 'day_of_week', 'is_weekend',
    'is_night', 'is_high_amount', 'is_extreme',
    'amount_band', 'night_x_amount', 'wifi_web'
]

# ── In-memory log ──────────────────────────────────────────
transaction_log = []


# ══════════════════════════════════════════════════════════
# PREPROCESSING — matches notebook Cell 4 exactly
# ══════════════════════════════════════════════════════════
def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Engineered features — same as notebook Cell 4
    df['is_night']       = int(data['hour_of_day'] <= 4)
    df['is_high_amount'] = int(data['amount (INR)'] > 1596)
    df['is_extreme']     = int(data['amount (INR)'] > 20000)
    df['amount_band']    = pd.cut(
        df['amount (INR)'],
        bins=[0, 500, 1500, 5000, 15000, 50000],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(4).astype(int)
    df['night_x_amount'] = df['is_night'] * data['amount (INR)']
    df['wifi_web']       = int(
        data['network_type'] == 'WiFi' and
        data['device_type'] == 'Web'
    )

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        le  = label_encoders[col]
        val = str(data[col])
        df[col] = le.transform([val])[0] if val in le.classes_ else 0

    # Ensure columns match scaler exactly
    df_features = df[FEATURE_COLS]

    # Scale ALL features
    scaled_values = scaler.transform(df_features)

    return pd.DataFrame(scaled_values, columns=FEATURE_COLS)


# ══════════════════════════════════════════════════════════
# PREDICTION — matches reshape from notebook Cell 11/12
# ══════════════════════════════════════════════════════════
def get_fraud_probability(features: pd.DataFrame) -> float:
    if MODEL_TYPE == 'keras':
        # Both LSTM models use reshape(-1, n_features, 1)
        # This matches: X_test.reshape(-1, n_features, 1) in notebook
        n_features = features.shape[1]
        X = features.values.reshape(1, n_features, 1)
        return float(model.predict(X, verbose=0)[0][0])

    elif MODEL_TYPE == 'kmeans':
        dists = model.transform(features)
        score = float(dists.min())
        return float(1 / (1 + np.exp(-0.5 * (score - 3.0))))

    else:
        return float(model.predict_proba(features)[0][1])


# ══════════════════════════════════════════════════════════
# RISK LEVEL
# ══════════════════════════════════════════════════════════
def get_risk_level(prob: float) -> str:
    if prob >= 0.70:            return 'HIGH'
    if prob >= 0.45:            return 'MEDIUM'
    if prob >= FRAUD_THRESHOLD: return 'LOW'
    return 'SAFE'


# ══════════════════════════════════════════════════════════
# MINIMAL BUSINESS RULES — nudges only, ML drives decisions
# ══════════════════════════════════════════════════════════
def apply_minimal_rules(data: dict, fraud_prob: float):
    amount   = float(data['amount (INR)'])
    hour     = int(data['hour_of_day'])
    network  = str(data['network_type'])
    txn_type = str(data['transaction type'])
    reasons  = []

    # Only two extreme edge-case nudges remain
    if amount > 50000 and hour <= 3 and network == 'WiFi':
        fraud_prob = min(1.0, fraud_prob + 0.12)
        reasons.append("Extreme amount >₹50k at midnight on WiFi")

    if txn_type == 'Recharge' and amount > 10000:
        fraud_prob = min(1.0, fraud_prob + 0.08)
        reasons.append("Unusually large recharge >₹10k")

    if not reasons:
        reasons.append("ML model only")

    return round(fraud_prob, 4), reasons


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':  'running',
        'message': 'UPI Fraud Detection API',
        'model':   MODEL_NAME,
        'endpoints': {
            'predict':     'POST /predict',
            'health':      'GET  /health',
            'model-info':  'GET  /model-info',
            'history':     'GET  /history',
            'stats':       'GET  /stats',
            'fraud-log':   'GET  /fraud-log',
            'fraud-stats': 'GET  /fraud-stats',
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'healthy',
        'model':     MODEL_NAME,
        'type':      MODEL_TYPE,
        'threshold': FRAUD_THRESHOLD
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'best_model': MODEL_NAME,
        'type':       MODEL_TYPE,
        'threshold':  FRAUD_THRESHOLD,
        'auc':        meta.get('auc', 'N/A'),
        'recall':     meta.get('recall', 'N/A'),
        'f1':         meta.get('f1', 'N/A'),
        'score':      meta.get('score', 'N/A'),
        'description': (
            "Best model selected from 6 algorithms: "
            "Random Forest, XGBoost, SVM, K-Means, "
            "Vanilla LSTM-CNN, Sequential LSTM-CNN with Attention."
        )
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body'}), 400

        required = [
            'transaction type', 'merchant_category', 'amount (INR)',
            'sender_age_group', 'receiver_age_group', 'sender_state',
            'sender_bank', 'receiver_bank', 'device_type',
            'network_type', 'hour_of_day', 'day_of_week', 'is_weekend'
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': 'Missing fields', 'missing': missing}), 400

        # Core ML prediction
        features   = preprocess(data)
        fraud_prob = get_fraud_probability(features)

        # Minimal rule nudge
        fraud_prob, triggered_rules = apply_minimal_rules(data, fraud_prob)

        is_fraud   = bool(fraud_prob >= FRAUD_THRESHOLD)
        risk_level = get_risk_level(fraud_prob)
        action     = 'BLOCK' if is_fraud else 'ALLOW'

        response = {
            'transaction_id':    data.get('transaction_id', 'N/A'),
            'amount':            data['amount (INR)'],
            'fraud_probability': fraud_prob,
            'is_fraud':          is_fraud,
            'risk_level':        risk_level,
            'action':            action,
            'message':           ('Fraudulent transaction detected!'
                                  if is_fraud
                                  else 'Transaction appears legitimate.'),
            'triggered_rules':   triggered_rules,
            'model_used':        MODEL_NAME
        }

        if is_fraud:
            log_fraud_case(data, response)
            if should_alert(response):
                send_fraud_alert(data, response)

        _log_transaction(data, response)

        print(f"[{MODEL_NAME}] ₹{data['amount (INR)']} "
              f"prob={fraud_prob:.3f} → {action} | "
              f"rules={triggered_rules}")

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    return jsonify(transaction_log), 200


@app.route('/stats', methods=['GET'])
def stats():
    total  = len(transaction_log)
    frauds = sum(1 for t in transaction_log if t['result']['is_fraud'])
    return jsonify({
        'total_transactions': total,
        'fraud_detected':     frauds,
        'legitimate':         total - frauds,
        'fraud_rate':         round(frauds / total * 100, 2) if total else 0
    })


@app.route('/fraud-log', methods=['GET'])
def fraud_log():
    cases = get_all_fraud_cases()
    return jsonify({'total': len(cases), 'cases': cases}), 200


@app.route('/fraud-stats', methods=['GET'])
def fraud_statistics():
    return jsonify(get_fraud_stats()), 200


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def _log_transaction(data, result):
    transaction_log.append({
        'amount':      data['amount (INR)'],
        'sender_bank': data['sender_bank'],
        'device':      data['device_type'],
        'hour':        data['hour_of_day'],
        'result':      result
    })
    if len(transaction_log) > 200:
        transaction_log.pop(0)


if __name__ == '__main__':
    app.run(debug=True, port=5000)