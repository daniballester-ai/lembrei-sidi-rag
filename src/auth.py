import hashlib
from typing import Optional, Dict

from src.data import load_json, save_json

USERS_FILE = "users.json"


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _seed_admin():
    users = load_json(USERS_FILE, [])
    if not any(u["email"] == "admin@sidi.org" for u in users):
        users.append({
            "email": "admin@sidi.org",
            "password_hash": _hash_password("admin123"),
            "role": "admin",
        })
        save_json(USERS_FILE, users)


def create_user(email: str, password: str, role: str = "student") -> bool:
    users = load_json(USERS_FILE, [])
    if any(u["email"] == email.lower() for u in users):
        return False
    users.append({
        "email": email.lower(),
        "password_hash": _hash_password(password),
        "role": role,
    })
    save_json(USERS_FILE, users)
    return True


def authenticate(email: str, password: str) -> Optional[Dict[str, str]]:
    users = load_json(USERS_FILE, [])
    pw_hash = _hash_password(password)
    for u in users:
        if u["email"] == email.lower() and u["password_hash"] == pw_hash:
            return {"email": u["email"], "role": u["role"]}
    return None


_seed_admin()
