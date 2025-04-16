import os
import hashlib
import hmac
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

class Security:
    def __init__(self, secret_key=None):
        self.secret_key = secret_key or os.urandom(32)

    def hash_password(self, password, salt=None):
        salt = salt or os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return base64.urlsafe_b64encode(salt + key).decode()

    def verify_password(self, password, hashed):
        decoded = base64.urlsafe_b64decode(hashed.encode())
        salt, key = decoded[:16], decoded[16:]
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        try:
            kdf.verify(password.encode(), key)
            return True
        except Exception:
            return False

    def encrypt_message(self, message):
        salt = os.urandom(16)
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
            backend=default_backend()
        )
        key = kdf.derive(self.secret_key)
        h = hmac.new(key, message.encode(), hashlib.sha256)
        return base64.urlsafe_b64encode(salt + h.digest()).decode()

    def verify_message(self, message, encrypted):
        decoded = base64.urlsafe_b64decode(encrypted.encode())
        salt, hmac_digest = decoded[:16], decoded[16:]
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
            backend=default_backend()
        )
        key = kdf.derive(self.secret_key)
        h = hmac.new(key, message.encode(), hashlib.sha256)
        return hmac.compare_digest(h.digest(), hmac_digest)

    def generate_auth_token(self, user_id):
        message = f"{user_id}:{os.urandom(16).hex()}"
        return self.encrypt_message(message)

    def verify_auth_token(self, token):
        try:
            decoded = base64.urlsafe_b64decode(token.encode())
            salt, hmac_digest = decoded[:16], decoded[16:]
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend()
            )
            key = kdf.derive(self.secret_key)
            h = hmac.new(key, token.encode(), hashlib.sha256)
            return hmac.compare_digest(h.digest(), hmac_digest)
        except Exception:
            return False
