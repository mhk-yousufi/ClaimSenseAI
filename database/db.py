# database/db.py
import sqlite3
import pandas as pd
import os

DB_PATH = "claim_sense.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_model TEXT,
            claim_amount REAL,
            driver_age INTEGER,
            previous_claims INTEGER,
            damage_severity TEXT,
            cnn_confidence REAL,
            fraud_score REAL,
            llm_summary TEXT,
            status TEXT DEFAULT 'Pending Review',
            modified_amount REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_claim(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO claims (vehicle_model, claim_amount, driver_age, previous_claims, damage_severity, cnn_confidence, fraud_score, llm_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['vehicle_model'], data['claim_amount'], data['driver_age'], data['previous_claims'], 
          data['damage_severity'], data['cnn_confidence'], data['fraud_score'], data['llm_summary']))
    claim_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return claim_id

def update_claim(claim_id, status, modified_amount):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE claims SET status = ?, modified_amount = ? WHERE claim_id = ?
    ''', (status, modified_amount, claim_id))
    conn.commit()
    conn.close()

def get_all_claims():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM claims", conn)
    conn.close()
    return df