"""Local server with a login page and scrypt password hashes."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / "data" / "auth.sqlite"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
SESSION_COOKIE = "tram_session"
SESSION_DAYS = 7
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024
SALT_BYTES = 16
PUBLIC_FILES = {"login.html", "styles.css"}
BLOCKED_SUFFIXES = {".py", ".sqlite", ".db", ".pem", ".env"}

db_lock = threading.Lock()
connection = None
dummy_salt = os.urandom(SALT_BYTES)
dummy_hash = hashlib.scrypt(b"\0", salt=dummy_salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN)


def now_utc():
    return datetime.now(timezone.utc)


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
        dklen=SCRYPT_DKLEN,
    )
    return salt, digest


def open_database():
    global connection
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            scrypt_n INTEGER NOT NULL,
            scrypt_r INTEGER NOT NULL,
            scrypt_p INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        );
        """
    )
    connection.commit()
    ensure_user("test", "test")


def ensure_user(username, password):
    with db_lock:
        existing = connection.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return
        salt, digest = hash_password(password)
        connection.execute(
            """
            INSERT INTO users (username, salt, password_hash, scrypt_n, scrypt_r, scrypt_p, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, salt, digest, SCRYPT_N, SCRYPT_R, SCRYPT_P, now_utc().isoformat()),
        )
        connection.commit()


def verify_password(username, password):
    with db_lock:
        row = connection.execute(
            "SELECT salt, password_hash, scrypt_n, scrypt_r, scrypt_p FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        digest = hashlib.scrypt(password.encode("utf-8"), salt=dummy_salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN)
        hmac.compare_digest(digest, dummy_hash)
        return False
    salt, stored, n_cost, r_cost, p_cost = row
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n_cost, r=r_cost, p=p_cost, maxmem=SCRYPT_MAXMEM, dklen=len(stored))
    return hmac.compare_digest(digest, stored)


def create_session(username):
    token = os.urandom(32).hex()
    expires = now_utc() + timedelta(days=SESSION_DAYS)
    with db_lock:
        connection.execute(
            "INSERT INTO sessions (token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, expires.isoformat()),
        )
        connection.commit()
    return token


def read_session(token):
    if not token or len(token) > 80:
        return None
    with db_lock:
        row = connection.execute(
            "SELECT username, expires_at FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        expires = datetime.fromisoformat(row[1])
        if expires < now_utc():
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            connection.commit()
            return None
    return row[0]


def delete_session(token):
    if not token:
        return
    with db_lock:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
        connection.commit()


def safe_next(value):
    if not value or not value.startswith("/") or value.startswith("//") or value.startswith("/\\"):
        return "/index.html"
    path = urlparse(value).path
    if path in {"/login", "/login.html", "/logout"}:
        return "/index.html"
    return value


class AppHandler(BaseHTTPRequestHandler):
    server_version = "TramAuth/1.0"

    def log_message(self, fmt, *args):
        super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/logout":
            self.handle_logout()
            return
        if path in {"/login", "/login.html"}:
            if self.current_user():
                self.redirect("/index.html")
                return
            self.serve_file(ROOT / "login.html")
            return
        if path in {"", "/"}:
            path = "/index.html"
        full = (ROOT / path.lstrip("/")).resolve()
        if self.public_file(full):
            self.serve_file(full)
            return
        if not self.current_user():
            target = quote(self.path, safe="/?=&")
            self.redirect(f"/login.html?next={target}")
            return
        if not self.allowed_file(full):
            self.send_error(404)
            return
        self.serve_file(full)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/logout":
            self.handle_logout()
            return
        if parsed.path != "/login":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length < 0 or length > 4096:
            self.send_error(413)
            return
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body, keep_blank_values=True)
        username = (form.get("login") or [""])[0].strip()
        password = (form.get("password") or [""])[0]
        destination = safe_next((form.get("next") or [""])[0])
        if username and password and verify_password(username, password):
            token = create_session(username)
            self.send_response(303)
            self.send_header("Location", destination)
            self.send_header("Set-Cookie", session_cookie(token))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.redirect("/login.html?error=1")

    def current_user(self):
        return read_session(self.cookies().get(SESSION_COOKIE))

    def cookies(self):
        found = {}
        for part in self.headers.get("Cookie", "").split(";"):
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            found[name.strip()] = value.strip()
        return found

    def handle_logout(self):
        delete_session(self.cookies().get(SESSION_COOKIE))
        self.send_response(303)
        self.send_header("Location", "/login.html")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT",
        )
        self.end_headers()

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def inside_root(self, full):
        try:
            full.relative_to(ROOT)
        except ValueError:
            return False
        if not full.is_file():
            return False
        relative = full.relative_to(ROOT)
        if any(part.startswith(".") for part in relative.parts):
            return False
        return full.suffix.lower() not in BLOCKED_SUFFIXES

    def public_file(self, full):
        return self.inside_root(full) and full.name in PUBLIC_FILES

    def allowed_file(self, full):
        return self.inside_root(full)

    def serve_file(self, full):
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type(full))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store" if full.name == "login.html" else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)


def session_cookie(token):
    max_age = SESSION_DAYS * 24 * 60 * 60
    return f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={max_age}"


def content_type(path):
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".geojson": "application/geo+json",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".zip": "application/zip",
        ".txt": "text/plain; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def main():
    open_database()
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    shown = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
    print(f"http://{shown}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        connection.close()


if __name__ == "__main__":
    main()
