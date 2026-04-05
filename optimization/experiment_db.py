import sqlite3
from datetime import datetime
import json

DB_PATH = "experiments.db"

_best_config_cache = None
_cache_timestamp = None

def get_db_connection():
    """Returns a SQLite connection optimized for concurrency."""
    # timeout=30 prevents "database is locked" errors under load
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level="IMMEDIATE")
    # WAL mode enables concurrent reads while writing
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # Drop table to ensure schema update given it's a test environment
    c.execute("DROP TABLE IF EXISTS experiments")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        question TEXT,
        config TEXT,
        answer_relevance REAL,
        faithfulness REAL
    )
    """)

    conn.commit()
    conn.close()


def log_experiment(question, config, answer_relevance, faithfulness):
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("""
    INSERT INTO experiments (timestamp, question, config, answer_relevance, faithfulness)
    VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        question,
        json.dumps(config),
        answer_relevance,
        faithfulness
    ))

    conn.commit()
    conn.close()


def get_best_config():

    global _best_config_cache

    if _best_config_cache:
        return _best_config_cache

    conn = get_db_connection()
    c = conn.cursor()

    c.execute("""
    SELECT config, AVG(answer_relevance + faithfulness)/2 as score
    FROM experiments
    GROUP BY config
    ORDER BY score DESC
    LIMIT 1
    """)

    row = c.fetchone()
    conn.close()

    if row:
        _best_config_cache = json.loads(row[0])
        return _best_config_cache

    return None