"""
database.py
===========
SQLite storage for analysed emergency messages.

The database file (emergency.db) and the table are created automatically the
first time init_db() runs, so no manual setup is needed.
"""

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "emergency.db")

HISTORY_COLUMNS = [
    "id", "message", "emergency_type", "priority", "location",
    "people_affected", "required_service", "confidence", "timestamp",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS emergency_messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    message          TEXT    NOT NULL,
    emergency_type   TEXT,
    priority         TEXT,
    location         TEXT,
    people_affected  TEXT,
    required_service TEXT,
    confidence       REAL,
    timestamp        TEXT
)
"""


def get_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite file (created if it does not exist)."""
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db() -> bool:
    """Create the emergency_messages table if needed. Returns True on success."""
    try:
        with closing(get_connection()) as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Database initialisation failed: %s", exc)
        return False


def save_message(result: Dict[str, object]) -> Optional[int]:
    """
    Save one analysis result. Returns the new row id, or None if saving failed.
    (A database problem must never crash the dashboard.)
    """
    try:
        init_db()
        with closing(get_connection()) as conn:
            cursor = conn.execute(
                """INSERT INTO emergency_messages
                   (message, emergency_type, priority, location, people_affected,
                    required_service, confidence, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(result.get("message", "")),
                    str(result.get("emergency_type", "Other")),
                    str(result.get("priority", "Low")),
                    str(result.get("location", "Not specified")),
                    str(result.get("people_affected", "Not mentioned")),
                    str(result.get("required_service", "")),
                    float(result.get("confidence", 0.0)),
                    str(result.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ),
            )
            conn.commit()
            return cursor.lastrowid
    except (sqlite3.Error, ValueError, TypeError) as exc:
        logger.error("Could not save message: %s", exc)
        return None


def get_history(limit: Optional[int] = None) -> pd.DataFrame:
    """Return saved messages (newest first) as a pandas DataFrame."""
    try:
        init_db()
        query = "SELECT * FROM emergency_messages ORDER BY id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with closing(get_connection()) as conn:
            return pd.read_sql_query(query, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        logger.error("Could not read history: %s", exc)
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def count_messages() -> int:
    """Number of analysed messages stored in the database."""
    try:
        init_db()
        with closing(get_connection()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM emergency_messages").fetchone()[0])
    except sqlite3.Error:
        return 0


def delete_all_messages() -> bool:
    """Delete the complete history. Returns True on success."""
    try:
        init_db()
        with closing(get_connection()) as conn:
            conn.execute("DELETE FROM emergency_messages")
            conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Could not clear history: %s", exc)
        return False
