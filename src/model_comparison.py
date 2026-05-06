"""
UPI Fraud Detection — Full Model Comparison
Compares: Random Forest, XGBoost, SVM, K-Means, Vanilla LSTM-CNN, Sequential LSTM-CNN
Selects the best model and saves it for the API.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve,
    precision_recall_fscore_support, accuracy_score
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten
from keras.utils import pad_sequences

# ── Paths ─────────────────────────────────────────────────────────────
BASE        = 'D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection'
DATA_PATH   = f'{BASE}/data/upi_transactions_2024.csv'
MODEL_DIR   = f'{BASE}/model'
RESULTS_DIR = f'{BASE}/results'

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature config ────────────────────────────────────────────────────
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
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 'is_high_amount'
]


# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
def load_and_preprocess():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Proxy user ID (no real user ID in dataset)
    df['user_proxy'] = (df['sender_state'] + '_' + df['sender_age_group']
                        + '_' + df['device_type'] + '_' + df['sender_bank'])

    # Feature engineering
    df['is_night']       = (df['hour_of_day'] <= 4).astype(int)
    df['is_high_amount'] = (df['amount (INR)'] > 1596).astype(int)

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Scale numerics
    scaler = StandardScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])

    X = df[FEATURE_COLS].values
    y = df['fraud_flag'].values

    # Temporal split 80/20
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Fraud rate (test): {y_test.mean()*100:.2f}%")

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders, scaler, split


# ══════════════════════════════════════════════════════════════════════
# 2. MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════
def build_vanilla_lstm_cnn(input_shape):
    """Single-transaction CNN → LSTM (no sequence context)."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='vanilla_lstm_cnn')
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model


def build_sequential_lstm_cnn(input_shape):
    """Sequence-aware CNN → LSTM (history of 5 transactions per user)."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='sequential_lstm_cnn')
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model


# ══════════════════════════════════════════════════════════════════════
# 3. SEQUENCE BUILDER (for Sequential LSTM-CNN)
# ══════════════════════════════════════════════════════════════════════
def build_sequences(df, X, SEQ_LEN=5):
    print("Building per-user transaction sequences...")
    user_history = {}
    X_seq = []
    for i in range(len(df)):
        user = df['user_proxy'].iloc[i]
        if user not in user_history:
            user_history[user] = []
        user_history[user].append(X[i])
        seq = user_history[user][-SEQ_LEN:]
        X_seq.append(seq)
    X_seq_padded = pad_sequences(
        X_seq, maxlen=SEQ_LEN, dtype='float32', padding='pre', truncating='pre'
    )
    return X_seq_padded


# ══════════════════════════════════════════════════════════════════════
# 4. EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════
def evaluate(name, y_test, probs, threshold=0.5):
    preds  = (probs >= threshold).astype(int)
    auc    = roc_auc_score(y_test, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='binary', zero_division=0
    )
    acc    = accuracy_score(y_test, preds)
    caught = int(((preds == 1) & (y_test == 1)).sum())
    total_f = int(y_test.sum())
    fp     = int(((preds == 1) & (y_test == 0)).sum())
    tn     = int(((preds == 0) & (y_test == 0)).sum())
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  {name}  (threshold={threshold})")
    print(f"{'='*55}")
    print(classification_report(y_test, preds,
          target_names=['Legitimate', 'Fraud']))
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Fraud caught: {caught}/{total_f}  |  FPR: {fpr_rate:.3f}")

    return {
        'Model': name,
        'ROC-AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'Accuracy': round(acc, 4),
        'FP Rate': round(fpr_rate, 4),
        'Fraud Caught': caught,
        'Total Fraud': total_f,
        'Threshold': threshold
    }


# ══════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════════════════
def plot_roc_curves(models_probs, y_test):
    plt.figure(figsize=(10, 7))
    colors = ['#E63946','#F4A261','#2A9D8F','#457B9D','#9B2226','#6A0572']
    for (name, probs), color in zip(models_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison — All 6 Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/combined_roc_auc.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/combined_roc_auc.png")


def plot_confusion_matrices(models_preds, y_test):
    n = len(models_preds)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.ravel()
    for idx, (name, preds) in enumerate(models_preds.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=axes[idx], cbar=False,
                    xticklabels=['Legit','Fraud'],
                    yticklabels=['Legit','Fraud'])
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle('Confusion Matrices — All 6 Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/combined_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR}/combined_confusion_matrices.png")


def plot_metrics_bar(df_results):
    metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df_results))
    width = 0.2
    colors = ['#457B9D','#2A9D8F','#E63946','#F4A261']
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df_results[metric], width, label=metric, color=color, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_results['Model'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison — Key Metrics', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/metrics_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# 6. BEST MODEL SELECTOR
# ══════════════════════════════════════════════════════════════════════
def select_best_model(df_results):
    """
    Scoring: 40% Recall (fraud detection), 30% ROC-AUC,
             20% F1-Score, 10% low FP Rate.
    Goal: maximize fraud detection, minimize missed cases.
    """
    df = df_results.copy()
    df['Score'] = (
        0.40 * df['Recall'] +
        0.30 * df['ROC-AUC'] +
        0.20 * df['F1-Score'] +
        0.10 * (1 - df['FP Rate'])
    )
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    print(f"\n{'*'*55}")
    print(f"  BEST MODEL: {best['Model']}")
    print(f"  Composite Score : {best['Score']:.4f}")
    print(f"  ROC-AUC         : {best['ROC-AUC']:.4f}")
    print(f"  Recall (fraud)  : {best['Recall']:.4f}")
    print(f"  F1-Score        : {best['F1-Score']:.4f}")
    print(f"  FP Rate         : {best['FP Rate']:.4f}")
    print(f"{'*'*55}")
    return df, best


# ══════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # ── Load data ──────────────────────────────────────────
    df, X, y, X_train, X_test, y_train, y_test, label_encoders, scaler, split = load_and_preprocess()

    all_results  = []
    models_probs = {}   # name → prob array
    models_preds = {}   # name → pred array
    trained_models = {} # name → model object

    # ── SMOTE for tree/SVM models ──────────────────────────
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # ── (A) Random Forest ──────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    res = evaluate('Random Forest', y_test, rf_probs, threshold=0.35)
    all_results.append(res)
    models_probs['Random Forest'] = rf_probs
    models_preds['Random Forest'] = (rf_probs >= 0.35).astype(int)
    trained_models['Random Forest'] = {'model': rf, 'type': 'sklearn',
                                        'threshold': 0.35, 'name': 'Random Forest'}

    # ── (B) XGBoost ────────────────────────────────────────
    print("\nTraining XGBoost...")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb.fit(X_train_sm, y_train_sm)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    res = evaluate('XGBoost', y_test, xgb_probs, threshold=0.30)
    all_results.append(res)
    models_probs['XGBoost'] = xgb_probs
    models_preds['XGBoost'] = (xgb_probs >= 0.30).astype(int)
    trained_models['XGBoost'] = {'model': xgb, 'type': 'sklearn',
                                  'threshold': 0.30, 'name': 'XGBoost'}

    # ── (C) SVM ────────────────────────────────────────────
    print("\nTraining SVM (RBF kernel, balanced)...")
    # Use subset for speed on large datasets
    subset = min(80000, len(X_train_sm))
    idx = np.random.choice(len(X_train_sm), subset, replace=False)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced',
              C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_sm[idx], y_train_sm[idx])
    svm_probs = svm.predict_proba(X_test)[:, 1]
    res = evaluate('SVM', y_test, svm_probs, threshold=0.40)
    all_results.append(res)
    models_probs['SVM'] = svm_probs
    models_preds['SVM'] = (svm_probs >= 0.40).astype(int)
    trained_models['SVM'] = {'model': svm, 'type': 'sklearn',
                              'threshold': 0.40, 'name': 'SVM'}

    # ── (D) K-Means (anomaly detection) ────────────────────
    print("\nTraining K-Means (anomaly scoring)...")
    # Train on legit transactions only — fraud = outlier
    X_legit = X_train[y_train == 0]
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    km.fit(X_legit)
    # Distance to nearest centroid = anomaly score
    dists = km.transform(X_test)
    km_scores = dists.min(axis=1)  # raw distances
    # Normalise to [0,1]
    km_probs = (km_scores - km_scores.min()) / (km_scores.max() - km_scores.min() + 1e-9)
    res = evaluate('K-Means', y_test, km_probs, threshold=0.50)
    all_results.append(res)
    models_probs['K-Means'] = km_probs
    models_preds['K-Means'] = (km_probs >= 0.50).astype(int)
    trained_models['K-Means'] = {'model': km, 'type': 'kmeans',
                                  'threshold': 0.50, 'name': 'K-Means'}

    # ── (E) Vanilla LSTM-CNN ───────────────────────────────
    print("\nTraining Vanilla LSTM-CNN...")
    X_train_v = X_train_sm.reshape((X_train_sm.shape[0], X_train_sm.shape[1], 1))
    X_test_v  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    vanilla = build_vanilla_lstm_cnn((X_train_v.shape[1], 1))
    vanilla.fit(X_train_v, y_train_sm, epochs=5, batch_size=256,
                validation_split=0.1,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc', mode='max', patience=2, restore_best_weights=True)],
                verbose=1)
    vanilla_probs = vanilla.predict(X_test_v, verbose=0).ravel()
    res = evaluate('Vanilla LSTM-CNN', y_test, vanilla_probs, threshold=0.45)
    all_results.append(res)
    models_probs['Vanilla LSTM-CNN'] = vanilla_probs
    models_preds['Vanilla LSTM-CNN'] = (vanilla_probs >= 0.45).astype(int)
    trained_models['Vanilla LSTM-CNN'] = {'model': vanilla, 'type': 'keras_vanilla',
                                           'threshold': 0.45, 'name': 'Vanilla LSTM-CNN'}

    # ── (F) Sequential LSTM-CNN ────────────────────────────
    print("\nTraining Sequential LSTM-CNN...")
    X_seq = build_sequences(df, X, SEQ_LEN=5)
    X_train_seq = X_seq[:split]
    X_test_seq  = X_seq[split:]

    total_t = len(y_train)
    neg_t, pos_t = (y_train == 0).sum(), (y_train == 1).sum()
    class_weight = {0: (1/neg_t)*(total_t/2.0), 1: (1/pos_t)*(total_t/2.0)}

    seq_model = build_sequential_lstm_cnn((X_train_seq.shape[1], X_train_seq.shape[2]))
    seq_model.fit(
        X_train_seq, y_train,
        epochs=10, batch_size=512,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=3, restore_best_weights=True)],
        verbose=1
    )
    seq_probs = seq_model.predict(X_test_seq, verbose=0).ravel()
    res = evaluate('Sequential LSTM-CNN', y_test, seq_probs, threshold=0.46)
    all_results.append(res)
    models_probs['Sequential LSTM-CNN'] = seq_probs
    models_preds['Sequential LSTM-CNN'] = (seq_probs >= 0.46).astype(int)
    trained_models['Sequential LSTM-CNN'] = {'model': seq_model, 'type': 'keras_seq',
                                              'threshold': 0.46, 'name': 'Sequential LSTM-CNN'}

    # ── Build results table ────────────────────────────────
    df_results = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print(df_results[[
        'Model','ROC-AUC','Precision','Recall','F1-Score','FP Rate','Fraud Caught'
    ]].to_string(index=False))
    print('='*70)

    # ── Select best model ──────────────────────────────────
    df_ranked, best = select_best_model(df_results)

    # ── Save plots ─────────────────────────────────────────
    print("\nGenerating plots...")
    plot_roc_curves(models_probs, y_test)
    plot_confusion_matrices(models_preds, y_test)
    plot_metrics_bar(df_results)

    # Save ranked results CSV
    df_ranked.to_csv(f'{RESULTS_DIR}/model_comparison_results.csv', index=False)
    print(f"Saved: {RESULTS_DIR}/model_comparison_results.csv")

    # ── Save best model artifacts ──────────────────────────
    best_name = best['Model']
    best_info = trained_models[best_name]
    print(f"\nSaving best model artifacts: {best_name}")

    if best_info['type'] == 'keras_vanilla':
        best_info['model'].save(f'{MODEL_DIR}/best_model.h5')
    elif best_info['type'] == 'keras_seq':
        best_info['model'].save(f'{MODEL_DIR}/best_model.keras')
        best_info['model'].save(f'{MODEL_DIR}/best_model.h5')
    elif best_info['type'] == 'kmeans':
        joblib.dump(best_info['model'], f'{MODEL_DIR}/best_model.pkl')
    else:
        joblib.dump(best_info['model'], f'{MODEL_DIR}/best_model.pkl')

    # Save all sklearn models individually too
    joblib.dump(rf,  f'{MODEL_DIR}/rf_v2.pkl')
    joblib.dump(xgb, f'{MODEL_DIR}/xgb_v2.pkl')
    joblib.dump(svm, f'{MODEL_DIR}/svm_v2.pkl')
    joblib.dump(km,  f'{MODEL_DIR}/kmeans_v2.pkl')
    vanilla.save(f'{MODEL_DIR}/vanilla_lstm_cnn.h5')
    seq_model.save(f'{MODEL_DIR}/sequential_lstm_cnn.h5')
    seq_model.save(f'{MODEL_DIR}/lstm_cnn_seq_model.keras')

    # Save preprocessing artifacts (shared)
    joblib.dump(label_encoders, f'{MODEL_DIR}/label_encoders_v2.pkl')
    joblib.dump(scaler,         f'{MODEL_DIR}/scaler_v2.pkl')

    # Meta file for API
    meta = {
        'name':      best_info['name'],
        'type':      ('keras' if 'keras' in best_info['type'] else
                      'kmeans' if best_info['type'] == 'kmeans' else 'sklearn'),
        'threshold': best_info['threshold'],
        'is_seq':    best_info['type'] == 'keras_seq',
        'score':     round(float(best['Score']), 4),
        'auc':       round(float(best['ROC-AUC']), 4),
        'recall':    round(float(best['Recall']), 4),
    }
    joblib.dump(meta, f'{MODEL_DIR}/best_model_meta.pkl')
    print(f"Saved meta: {meta}")

    print(f"\n✅ DONE — Best model: {best_name}")
    print(f"   All artifacts saved to: {MODEL_DIR}")
    print(f"   All plots saved to:     {RESULTS_DIR}")


if __name__ == '__main__':
    main()
