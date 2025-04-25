import os
from cryptography.fernet import Fernet
import hashlib
import hmac
import base64
import jwt
import ssl
import socket

class Security:
    def __init__(self):
        self.key = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher = Fernet(self.key)
        self.hmac_key = os.environ.get("HMAC_KEY", base64.urlsafe_b64encode(os.urandom(32)))
        self.jwt_secret = os.environ.get("JWT_SECRET", base64.urlsafe_b64encode(os.urandom(32)))

    def encrypt(self, data):
        return self.cipher.encrypt(data.encode())

    def decrypt(self, token):
        return self.cipher.decrypt(token).decode()

    def authenticate(self, token, valid_token):
        return token == valid_token

    def generate_token(self, data):
        return self.cipher.encrypt(data.encode())

    def validate_token(self, token):
        try:
            self.cipher.decrypt(token)
            return True
        except:
            return False

    def encrypt_file(self, file_path):
        with open(file_path, 'rb') as file:
            encrypted_data = self.cipher.encrypt(file.read())
        with open(file_path + '.enc', 'wb') as file:
            file.write(encrypted_data)

    def decrypt_file(self, file_path):
        with open(file_path, 'rb') as file:
            decrypted_data = self.cipher.decrypt(file.read())
        with open(file_path.replace('.enc', ''), 'wb') as file:
            file.write(decrypted_data)

    def generate_hmac(self, message):
        return hmac.new(self.hmac_key, message.encode(), hashlib.sha256).hexdigest()

    def verify_hmac(self, message, hmac_to_verify):
        generated_hmac = self.generate_hmac(message)
        return hmac.compare_digest(generated_hmac, hmac_to_verify)

    def encrypt_with_hmac(self, data):
        encrypted_data = self.encrypt(data)
        hmac_value = self.generate_hmac(encrypted_data)
        return encrypted_data, hmac_value

    def decrypt_with_hmac(self, encrypted_data, hmac_value):
        if self.verify_hmac(encrypted_data, hmac_value):
            return self.decrypt(encrypted_data)
        else:
            raise ValueError("HMAC verification failed. Data integrity compromised.")

    def generate_jwt(self, payload):
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def verify_jwt(self, token):
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError("JWT has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid JWT")

    def create_ssl_context(self, certfile, keyfile):
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile, keyfile)
        return context

    def secure_socket(self, sock, context):
        return context.wrap_socket(sock, server_side=True)

    def encrypt_data(self, data):
        """Encrypt data using Fernet encryption."""
        return self.cipher.encrypt(data.encode())

    def decrypt_data(self, encrypted_data):
        """Decrypt data using Fernet encryption."""
        return self.cipher.decrypt(encrypted_data).decode()

    def generate_hmac(self, message):
        """Generate HMAC for a given message."""
        return hmac.new(self.hmac_key, message.encode(), hashlib.sha256).hexdigest()

    def verify_hmac(self, message, hmac_to_verify):
        """Verify HMAC for a given message."""
        generated_hmac = self.generate_hmac(message)
        return hmac.compare_digest(generated_hmac, hmac_to_verify)

    def generate_jwt(self, payload):
        """Generate JWT for a given payload."""
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def verify_jwt(self, token):
        """Verify JWT and return the decoded payload."""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError("JWT has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid JWT")

    def create_ssl_context(self, certfile, keyfile):
        """Create SSL context for secure communication."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile, keyfile)
        return context

    def secure_socket(self, sock, context):
        """Wrap socket with SSL context for secure communication."""
        return context.wrap_socket(sock, server_side=True)

    def encrypt_file(self, file_path):
        """Encrypt a file using Fernet encryption."""
        with open(file_path, 'rb') as file:
            encrypted_data = self.cipher.encrypt(file.read())
        with open(file_path + '.enc', 'wb') as file:
            file.write(encrypted_data)

    def decrypt_file(self, file_path):
        """Decrypt a file using Fernet encryption."""
        with open(file_path, 'rb') as file:
            decrypted_data = self.cipher.decrypt(file.read())
        with open(file_path.replace('.enc', ''), 'wb') as file:
            file.write(decrypted_data)

    def encrypt_with_hmac(self, data):
        """Encrypt data and generate HMAC for integrity verification."""
        encrypted_data = self.encrypt_data(data)
        hmac_value = self.generate_hmac(encrypted_data)
        return encrypted_data, hmac_value

    def decrypt_with_hmac(self, encrypted_data, hmac_value):
        """Decrypt data and verify HMAC for integrity verification."""
        if self.verify_hmac(encrypted_data, hmac_value):
            return self.decrypt_data(encrypted_data)
        else:
            raise ValueError("HMAC verification failed. Data integrity compromised.")
