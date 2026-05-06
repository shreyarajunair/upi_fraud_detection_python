import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input

def create_lstm_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # CNN Part for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM Part for sequence learning (we treat features as a pseudo-sequence here)
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        # Fully connected layers
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def main():
    print("Loading data...")
    df = pd.read_csv('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/data/upi_transactions_2024.csv')
    df = df.drop(columns=['transaction id', 'timestamp', 'transaction_status'])
    
    # Feature Engineering
    df['is_night'] = (df['hour_of_day'] <= 4).astype(int)
    df['is_high_amount'] = (df['amount (INR)'] > 1596).astype(int)
    df['amount_band'] = pd.cut(df['amount (INR)'], bins=[0, 500, 1500, 5000, 50000], labels=[0, 1, 2, 3]).astype(int)
    
    # Label Encoding
    categorical_cols = [
        'transaction type', 'merchant_category', 'sender_age_group',
        'receiver_age_group', 'sender_state', 'sender_bank',
        'receiver_bank', 'device_type', 'network_type', 'day_of_week'
    ]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    X = df.drop(columns=['fraud_flag'])
    y = df['fraud_flag']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scale_cols = ['amount (INR)', 'hour_of_day']
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # Reshape for LSTM-CNN (samples, time steps, features)
    # We treat each feature as a "time step" for demonstration
    X_train_res = X_train_sm.to_numpy().reshape((X_train_sm.shape[0], X_train_sm.shape[1], 1))
    X_test_res = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"X_train reshaped: {X_train_res.shape}")
    print(f"X_test reshaped: {X_test_res.shape}")
    
    # Model building
    model = create_lstm_cnn_model((X_train_res.shape[1], 1))
    model.summary()
    
    # Training
    print("Training LSTM-CNN model...")
    # Using a small number of epochs for demonstration
    history = model.fit(X_train_res, y_train_sm, epochs=3, batch_size=256, validation_split=0.2)
    
    # Evaluation
    print("Evaluating model...")
    y_prob = model.predict(X_test_res).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("\n" + "="*50)
    print("  LSTM-CNN (threshold=0.5)")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    
    caught = int(((y_pred == 1) & (y_test == 1)).sum())
    total = int((y_test == 1).sum())
    print(f"Fraud caught: {caught} / {total}")
    
if __name__ == "__main__":
    main()
