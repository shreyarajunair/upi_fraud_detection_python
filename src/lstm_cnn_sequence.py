import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.utils import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Input

def create_sequence_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # 1D CNN for local feature extraction across time steps
        Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.2),
        
        # LSTM for learning sequential dependencies over time
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def main():
    print("Loading data...")
    df = pd.read_csv('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/data/upi_transactions_2024.csv')
    
    # Sort chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['timestamp']).reset_index(drop=True)
    
    # Create a proxy for 'User ID' using demographic/device features
    df['user_proxy'] = df['sender_state'] + '_' + df['sender_age_group'] + '_' + df['device_type'] + '_' + df['sender_bank']
    
    # Drop columns not used as direct features
    df_features = df.drop(columns=['transaction id', 'timestamp', 'transaction_status', 'user_proxy'])
    
    # Feature Engineering
    df_features['is_night'] = (df_features['hour_of_day'] <= 4).astype(int)
    df_features['is_high_amount'] = (df_features['amount (INR)'] > 1596).astype(int)
    
    # Encode Categoricals
    categorical_cols = [
        'transaction type', 'merchant_category', 'sender_age_group',
        'receiver_age_group', 'sender_state', 'sender_bank',
        'receiver_bank', 'device_type', 'network_type', 'day_of_week'
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        label_encoders[col] = le
        
    # Scale Numericals
    scale_cols = ['amount (INR)', 'hour_of_day']
    scaler = StandardScaler()
    df_features[scale_cols] = scaler.fit_transform(df_features[scale_cols])
    
    # Add grouping columns back for sequence creation
    df_features['user_proxy'] = df['user_proxy']
    target = df_features['fraud_flag'].values
    df_features = df_features.drop(columns=['fraud_flag'])
    
    print("Generating sequences...")
    # We will use a sequence length of 5 (current transaction + 4 previous)
    SEQ_LEN = 5
    features_np = df_features.drop(columns=['user_proxy']).values
    
    # Dictionary to keep track of previous transactions per user
    user_history = {}
    
    X_seq = []
    
    for i in range(len(df_features)):
        user = df_features['user_proxy'].iloc[i]
        curr_feat = features_np[i]
        
        if user not in user_history:
            user_history[user] = []
            
        user_history[user].append(curr_feat)
        
        # Take the last SEQ_LEN transactions
        seq = user_history[user][-SEQ_LEN:]
        X_seq.append(seq)
        
    # Pad sequences to ensure uniform shape (SEQ_LEN, num_features)
    # dtype='float32' is important for neural networks
    X_seq_padded = pad_sequences(X_seq, maxlen=SEQ_LEN, dtype='float32', padding='pre', truncating='pre')
    y = target
    
    print(f"Sequence data shape: {X_seq_padded.shape}")
    
    # Train/Test Split (Temporal split: last 20% is test)
    split_idx = int(len(X_seq_padded) * 0.8)
    X_train, X_test = X_seq_padded[:split_idx], X_seq_padded[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Class weights for handling imbalance (No SMOTE needed for sequences)
    neg = len(y_train) - sum(y_train)
    pos = sum(y_train)
    total = len(y_train)
    
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class weights: {class_weight}")
    
    # Create and train model
    model = create_sequence_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    print("Training sequence LSTM-CNN model...")
    # Using EarlyStopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=512, 
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[early_stop]
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_prob = model.predict(X_test).ravel()
    
    # Threshold tuning
    best_threshold = 0.46
    y_pred = (y_prob >= best_threshold).astype(int)
    
    print("\n" + "="*50)
    print(f"  Sequential LSTM-CNN (threshold={best_threshold})")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    
    caught = int(((y_pred == 1) & (y_test == 1)).sum())
    total_fraud = int((y_test == 1).sum())
    print(f"Fraud caught: {caught} / {total_fraud}")
    
    # --- Generate and Save Plots ---
    print("\nGenerating evaluation plots...")
    os.makedirs('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/results', exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    plt.title('LSTM-CNN Sequence Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/results/dl_confusion_matrix.png')
    plt.close()
    
    # 2. ROC-AUC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - LSTM-CNN')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/results/dl_roc_auc_curve.png')
    plt.close()
    print("Plots saved in the 'results' folder!")
    
    # Save the model and preprocessing objects for the API/Dashboard
    print("\nSaving model and preprocessing artifacts...")
    os.makedirs('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/model', exist_ok=True)
    model.save('D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/model/lstm_cnn_seq_model.keras')
    import joblib
    joblib.dump(label_encoders, 'D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/model/seq_label_encoders.pkl')
    joblib.dump(scaler, 'D:/Shreya/SEM 2/MINOR PROJECT/upi-fraud-detection/model/seq_scaler.pkl')
    print("Saved successfully!")

if __name__ == "__main__":
    main()
