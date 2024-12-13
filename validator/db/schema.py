from typing import List
import sqlite3
from pathlib import Path
from datetime import datetime

SCHEMA_VERSION = 1

def get_schema_v1() -> List[str]:
    """Database schema for version 1"""
    return [
        # Challenges table
        """
        CREATE TABLE IF NOT EXISTS challenges (
            challenge_id INTEGER PRIMARY KEY,  -- This is the task_id from the API
            type TEXT NOT NULL,
            video_url TEXT,
            created_at TIMESTAMP NOT NULL,
            metadata JSON,
            task_type TEXT,
            task_name TEXT
        )
        """,
        
        # Challenge assignments table
        """
        CREATE TABLE IF NOT EXISTS challenge_assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id INTEGER NOT NULL,
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT CHECK(status IN ('assigned', 'sent', 'completed', 'failed')) DEFAULT 'assigned',
            FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id),
            UNIQUE(challenge_id, miner_hotkey)
        )
        """,
        
        # Responses table
        """
        CREATE TABLE IF NOT EXISTS responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id INTEGER NOT NULL,
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER,
            processing_time FLOAT,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            evaluated BOOLEAN DEFAULT FALSE,
            score FLOAT,
            evaluated_at TIMESTAMP,
            response_data JSON,
            FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id),
            FOREIGN KEY (challenge_id, miner_hotkey) REFERENCES challenge_assignments(challenge_id, miner_hotkey)
        )
        """,
        
        # Availability checks table
        """
        CREATE TABLE IF NOT EXISTS availability_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER NOT NULL,
            hotkey TEXT NOT NULL,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_available BOOLEAN NOT NULL,
            response_time_ms FLOAT NOT NULL,
            error TEXT
        )
        """,
        
        # Frame evaluations table
        """
        CREATE TABLE IF NOT EXISTS frame_evaluations (
            evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id INTEGER NOT NULL,
            challenge_id INTEGER NOT NULL,
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER NOT NULL,
            frame_id INTEGER NOT NULL,
            frame_timestamp FLOAT NOT NULL,
            frame_score FLOAT NOT NULL,
            raw_frame_path TEXT,
            annotated_frame_path TEXT,
            vlm_response JSON,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (response_id) REFERENCES responses(response_id),
            FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id),
            FOREIGN KEY (challenge_id, miner_hotkey) REFERENCES challenge_assignments(challenge_id, miner_hotkey)
        )
        """,
        
        # Response scores table
        """
        CREATE TABLE IF NOT EXISTS response_scores (
            score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id INTEGER NOT NULL,
            challenge_id INTEGER NOT NULL,
            miner_hotkey TEXT NOT NULL,
            validator_hotkey TEXT,
            evaluation_score FLOAT NOT NULL,
            availability_score FLOAT NOT NULL,
            speed_score FLOAT NOT NULL,
            total_score FLOAT NOT NULL,
            speed_scoring_factor FLOAT,
            response_time FLOAT,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (response_id) REFERENCES responses(response_id),
            FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id),
            FOREIGN KEY (challenge_id, miner_hotkey) REFERENCES challenge_assignments(challenge_id, miner_hotkey)
        )
        """
    ]

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize the database with all tables."""
    conn = sqlite3.connect(db_path)
    for query in get_schema_v1():
        conn.execute(query)
    conn.commit()
    return conn