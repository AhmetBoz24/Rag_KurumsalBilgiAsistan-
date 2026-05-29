"""Admin sifresini olusturup .env'e yazan tek seferlik kurulum scripti.

Kullanim:
    python scripts/setup_admin.py            # rastgele 16 karakter sifre
    python scripts/setup_admin.py mypass     # belirli sifre

.env'e yazar:
  ADMIN_USERNAME=admin
  ADMIN_PASSWORD_HASH=<bcrypt>
  JWT_SECRET=<random>
"""
import os
import secrets
import string
import sys
from pathlib import Path

# repo root'a path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auth import hash_password


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def random_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def random_secret(length: int = 48) -> str:
    return secrets.token_urlsafe(length)


def read_env() -> dict:
    if not ENV_PATH.exists():
        return {}
    out = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def write_env(env: dict) -> None:
    lines = [f"{k}={v}" for k, v in env.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    password = sys.argv[1] if len(sys.argv) > 1 else random_password()
    username = os.getenv("ADMIN_USERNAME", "admin")

    env = read_env()
    env["ADMIN_USERNAME"] = username
    env["ADMIN_PASSWORD_HASH"] = hash_password(password)

    if not env.get("JWT_SECRET"):
        env["JWT_SECRET"] = random_secret()

    if not env.get("GROQ_API_KEY"):
        env["GROQ_API_KEY"] = ""  # placeholder

    write_env(env)

    print("=" * 60)
    print("ADMIN KIMLIGI OLUSTURULDU")
    print("=" * 60)
    print(f"  Kullanici adi : {username}")
    print(f"  Sifre         : {password}")
    print()
    print("BU SIFREYI HEMEN GUVENLI BIR YERE KAYDET.")
    print("Bir daha gosterilmeyecek (sadece bcrypt hash'i .env'de).")
    print("=" * 60)
    if not env.get("GROQ_API_KEY"):
        print("UYARI: .env'deki GROQ_API_KEY bos. Manuel doldur.")


if __name__ == "__main__":
    main()
