
import os
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash


def get_mysql_config():
    """Build the MySQL configuration from environment variables."""
    host = os.getenv("SMARTPM_DB_HOST", "localhost")
    port = int(os.getenv("SMARTPM_DB_PORT", "3306"))
    user = os.getenv("SMARTPM_DB_USER", "smartpm_app")
    password = os.getenv("SMARTPM_DB_PASSWORD", "")
    database = os.getenv("SMARTPM_DB_NAME", "smartpm_db")

    if not password:
        raise RuntimeError(
            "MySQL password is not configured. "
            "Set the SMARTPM_DB_PASSWORD environment variable."
        )

    config = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "connection_timeout": 15,
    }

    # Require a trusted CA certificate for remote database connections.
    if host not in ("localhost", "127.0.0.1", "::1"):
        ca_path = os.getenv("SMARTPM_DB_SSL_CA")

        if not ca_path:
            raise RuntimeError(
                "Cloud MySQL requires a CA certificate. "
                "Set SMARTPM_DB_SSL_CA to the certificate file path."
            )

        if not os.path.isfile(ca_path):
            raise RuntimeError(
                "The MySQL CA certificate file was not found. "
                "Check SMARTPM_DB_SSL_CA."
            )

        config.update({
            "ssl_ca": ca_path,
            "ssl_verify_cert": True,
            "ssl_verify_identity": True,
        })

    return config


def get_connection():
    """Create a secure connection to the SmartPM MySQL database."""
    return mysql.connector.connect(**get_mysql_config())


def init_db():
    """Create the users table if it does not already exist."""
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(80) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
                approved TINYINT(1) NOT NULL DEFAULT 0
            )
        """)
        connection.commit()
        print("MySQL authentication database initialized successfully.")

    finally:
        cursor.close()
        connection.close()


def create_user(username, password, role="user", approved=False):
    username = username.strip()

    if not username or len(username) > 80:
        raise ValueError("Username must contain 1–80 characters.")

    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")

    if role not in ("admin", "user"):
        raise ValueError("Invalid user role.")

    password_hash = generate_password_hash(password)
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, role, approved)
            VALUES (%s, %s, %s, %s)
            """,
            (username, password_hash, role, int(approved)),
        )
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_user_by_username(username):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT * FROM users WHERE username = %s",
            (username.strip(),),
        )
        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()


def get_user_by_id(user_id):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT * FROM users WHERE id = %s",
            (user_id,),
        )
        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()


def verify_password(user, password):
    return bool(
        user
        and check_password_hash(user["password_hash"], password)
    )


if __name__ == "__main__":
    init_db()
