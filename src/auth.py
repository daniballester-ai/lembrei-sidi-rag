import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Dict

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "users.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    conn.commit()
    
    # Criar admin padrao se nao existir
    cursor.execute("SELECT * FROM users WHERE email = ?", ("admin@sidi.org",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
            ("admin@sidi.org".lower(), _hash_password("admin123"), "admin")
        )
        conn.commit()
        
    conn.close()

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email: str, password: str, role: str = "student") -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
            (email.lower(), _hash_password(password), role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate(email: str, password: str) -> Optional[Dict[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT email, role FROM users WHERE email = ? AND password_hash = ?",
        (email.lower(), _hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {"email": user[0], "role": user[1]}
    return None

# Inicializa o banco de dados e o admin default na importação do módulo
init_db()
