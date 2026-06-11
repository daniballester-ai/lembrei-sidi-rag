from __future__ import annotations

import json
import os
import base64
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or None


def _github_repo() -> str | None:
    return os.environ.get("GITHUB_REPO") or None


def load_json(filename: str, default: list | None = None) -> any:
    path = DATA_DIR / filename
    if not path.exists():
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename: str, data: any) -> None:
    _ensure_data_dir()
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _sync_to_github(filename)


def _sync_to_github(filename: str) -> bool:
    token = _github_token()
    repo = _github_repo()
    if not token or not repo:
        return False

    path = DATA_DIR / filename
    if not path.exists():
        return False

    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    api_url = f"https://api.github.com/repos/{repo}/contents/data/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    sha = None
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.ok:
            sha = resp.json().get("sha")
    except requests.RequestException:
        pass

    payload = {
        "message": f"sync(data): atualizar {filename}",
        "content": content,
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        return resp.ok
    except requests.RequestException:
        return False


def fetch_from_github(filename: str) -> bool:
    token = _github_token()
    repo = _github_repo()
    if not token or not repo:
        return False

    api_url = f"https://api.github.com/repos/{repo}/contents/data/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if not resp.ok:
            return False
        data = resp.json()
        raw = base64.b64decode(data["content"]).decode("utf-8")
        path = DATA_DIR / filename
        _ensure_data_dir()
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
        return True
    except Exception:
        return False


def init_data():
    _ensure_data_dir()
    for filename in ["users.json", "deadlines.json"]:
        path = DATA_DIR / filename
        if not path.exists():
            fetch_from_github(filename)


init_data()
