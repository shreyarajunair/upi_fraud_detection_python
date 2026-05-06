import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fraud_log.db')

def init_db():
    """
    Creates the database and fraud_cases table
    if they don't already exist.
    Called once when the Flask app starts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            amount              REAL NOT NULL,
            transaction_type    TEXT,
            sender_bank         TEXT,
            receiver_bank       TEXT,
            sender_state        TEXT,
            device_type         TEXT,
            network_type        TEXT,
            hour_of_day         INTEGER,
            is_weekend          INTEGER,
            fraud_probability   REAL,
            risk_level          TEXT,
            action              TEXT,
            alerted             INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized at:", DB_PATH)


def log_fraud_case(data, result):
    """
    Saves a flagged fraud transaction to the database.
    Called only when is_fraud=True.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO fraud_cases (
            timestamp, amount, transaction_type,
            sender_bank, receiver_bank, sender_state,
            device_type, network_type, hour_of_day,
            is_weekend, fraud_probability, risk_level,
            action, alerted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data['amount (INR)'],
        data['transaction type'],
        data['sender_bank'],
        data['receiver_bank'],
        data['sender_state'],
        data['device_type'],
        data['network_type'],
        data['hour_of_day'],
        data['is_weekend'],
        result['fraud_probability'],
        result['risk_level'],
        result['action'],
        1  # alerted = True
    ))

    conn.commit()
    conn.close()


def get_all_fraud_cases():
    """
    Returns all fraud cases from the database.
    Used by the /fraud-log endpoint and dashboard.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Returns rows as dicts
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM fraud_cases
        ORDER BY timestamp DESC
    ''')

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_fraud_stats():
    """
    Returns summary statistics for the dashboard.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM fraud_cases')
    total_fraud = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(fraud_probability) FROM fraud_cases')
    avg_prob = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(amount) FROM fraud_cases')
    avg_amount = cursor.fetchone()[0]

    cursor.execute('''
        SELECT risk_level, COUNT(*) as count
        FROM fraud_cases
        GROUP BY risk_level
    ''')
    by_risk = dict(cursor.fetchall())

    cursor.execute('''
        SELECT transaction_type, COUNT(*) as count
        FROM fraud_cases
        GROUP BY transaction_type
        ORDER BY count DESC
    ''')
    by_type = dict(cursor.fetchall())

    conn.close()

    return {
        'total_fraud_cases':      total_fraud,
        'avg_fraud_probability':  round(avg_prob or 0, 4),
        'avg_fraud_amount':       round(avg_amount or 0, 2),
        'by_risk_level':          by_risk,
        'by_transaction_type':    by_type
    }