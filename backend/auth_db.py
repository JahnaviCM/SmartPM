
import os
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash


# Database configuration.
# Local defaults preserve your existing local SmartPM setup.
# On Render, configure these values using Environment Variables.
MYSQL_CONFIG = {
    "host": os.getenv("SMARTPM_DB_HOST", "localhost"),
    "port": int(os.getenv("SMARTPM_DB_PORT", "3306")),
    "user": os.getenv("SMARTPM_DB_USER", "smartpm_app"),
    "password": os.getenv("SMARTPM_DB_PASSWORD", ""),
    "database": os.getenv("SMARTPM_DB_NAME", "smartpm_db"),
    "connection_timeout": 15,
}


def get_connection():
    """Connect to MySQL, verifying TLS for cloud database connections."""

    if not MYSQL_CONFIG["password"]:
        raise RuntimeError(
            "MySQL password is not configured. "
            "Set SMARTPM_DB_PASSWORD."
        )

    config = MYSQL_CONFIG.copy()
    host = config["host"].lower()

    # Cloud database connections must use a trusted CA certificate.
    if host not in ("localhost", "127.0.0.1", "::1"):
        ssl_ca = os.getenv("SMARTPM_DB_SSL_CA")

        if not ssl_ca:
            raise RuntimeError(
                "Cloud MySQL requires a CA certificate. "
                "Configure SMARTPM_DB_SSL_CA in your environment."
            )

        if not os.path.isfile(ssl_ca):
            raise RuntimeError(
                f"MySQL CA certificate was not found at: {ssl_ca}"
            )

        config.update({
            "ssl_ca": ssl_ca,
            "ssl_verify_cert": True,
            "ssl_verify_identity": True,
        })

    return mysql.connector.connect(
    pool_name="smartpm_pool",
    pool_size=5,
    **config
)


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

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def create_user(username, password, role="user", approved=False):
    """Create a user with a securely hashed password."""

    if not isinstance(username, str):
        raise ValueError("Username must be text.")

    username = username.strip()

    if not username or len(username) > 80:
        raise ValueError("Username must contain 1–80 characters.")

    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")

    if len(password) > 1024:
        raise ValueError("Password is too long.")

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
            (username, password_hash, role, int(bool(approved))),
        )
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_user_by_username(username):
    """Return a user record by username, or None if not found."""

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
    """Return a user record by ID, or None if not found."""

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
    """Verify a submitted password against the stored password hash."""

    return bool(
        user
        and isinstance(password, str)
        and check_password_hash(user["password_hash"], password)
    )


if __name__ == "__main__":
    init_db()
