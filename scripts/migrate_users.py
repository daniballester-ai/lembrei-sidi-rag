"""Migra users.db (SQLite) para users.json.

Uso: python scripts/migrate_users.py
"""
import json
import sqlite3
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "users.db"
JSON_PATH = ROOT / "data" / "users.json"


def migrate():
    if not DB_PATH.exists():
        print("users.db nao encontrado. Nada a migrar.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT email, password_hash, role FROM users")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("users.db vazio. Nada a migrar.")
        return

    users = [
        {"email": email, "password_hash": pw, "role": role}
        for email, pw, role in rows
    ]

    # Hash das senhas se estiverem em texto puro
    for u in users:
        if len(u["password_hash"]) != 64:
            u["password_hash"] = hashlib.sha256(u["password_hash"].encode()).hexdigest()

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    print(f"Migrados {len(users)} usuarios para {JSON_PATH}")


if __name__ == "__main__":
    migrate()
