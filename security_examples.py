import re
import html
import logging
from cryptography.fernet import Fernet

# Input Validation Example
def validate_input(input_string):
    # Check if the input string contains any special characters
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', input_string):
        raise ValueError("Input contains special characters.")

# Output Encoding Example
def encode_output(output_string):
    # Encode the output string to prevent XSS attacks
    return html.escape(output_string)

# Encryption Example
def encrypt_data(data):
    # Generate a key
    key = Fernet.generate_key()

    # Create a Fernet cipher
    cipher = Fernet(key)

    # Encrypt the data
    encrypted_data = cipher.encrypt(data.encode())

    return encrypted_data, key

def decrypt_data(encrypted_data, key):
    # Create a Fernet cipher
    cipher = Fernet(key)

    # Decrypt the data
    decrypted_data = cipher.decrypt(encrypted_data).decode()

    return decrypted_data

# Logging and Monitoring Example
def log_activity(activity):
    # Log the activity
    logging.info(activity)

if __name__ == "__main__":
    # Input Validation Example
    try:
        validate_input("Hello, World!")
    except ValueError as e:
        print(e)

    # Output Encoding Example
    encoded_output = encode_output("<script>alert('Hello, World!')</script>")
    print(encoded_output)

    # Encryption Example
    data = "Hello, World!"
    encrypted_data, key = encrypt_data(data)
    decrypted_data = decrypt_data(encrypted_data, key)
    print(decrypted_data)

    # Logging and Monitoring Example
    log_activity("User logged in.") 