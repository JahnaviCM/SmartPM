
import os
import sqlite3
import getpass
import mysql.connector

SQLITE_PATH = os.path.join("backend", "users.db")

MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "smartpm_app",
    "database": "smartpm_db",
}

TABLE_COLUMNS = {
    "users": [
        "id", "username", "password_hash", "role", "approved"
    ],
    "login_attempts": [
        "username", "failed_count", "first_failed", "locked_until"
    ],
    "prediction_history": [
        "id", "user_id", "created_at", "machine_type",
        "air_temperature", "process_temperature",
        "rotational_speed", "torque", "tool_wear",
        "result", "failure_probability", "risk_level",
        "threshold", "recommendation"
    ],
}


def main():
    if not os.path.isfile(SQLITE_PATH):
        print(f"ERROR: Database not found: {SQLITE_PATH}")
        print("Run this script from your SmartPM project root.")
        return

    password = getpass.getpass(
        "Enter the MySQL password for smartpm_app: "
    )

    config = dict(MYSQL_CONFIG)
    config["password"] = password

    source = None
    target = None
    cursor = None
    transaction_started = False

    try:
        # Open the original SQLite database in read-only mode.
        source_uri = (
            "file:"
            + os.path.abspath(SQLITE_PATH).replace("\\", "/")
            + "?mode=ro"
        )
        source = sqlite3.connect(source_uri, uri=True)
        source.row_factory = sqlite3.Row

        # Connect to MySQL.
        target = mysql.connector.connect(**config)
        cursor = target.cursor()

        print("\nConnected to MySQL successfully.")

        # Read all source records first.
        source_data = {}

        for table, columns in TABLE_COLUMNS.items():
            column_sql = ", ".join(
                f'"{column}"' for column in columns
            )
            rows = source.execute(
                f"SELECT {column_sql} FROM `{table}`"
            ).fetchall()

            source_data[table] = [
                tuple(row[column] for column in columns)
                for row in rows
            ]

        print("\nSource SQLite record counts:")
        for table, rows in source_data.items():
            print(f"  {table}: {len(rows)}")

        # Ensure the MySQL destination tables are empty.
        # End any transaction started by these SELECT statements.
        for table in TABLE_COLUMNS:
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            count = cursor.fetchone()[0]

            if count != 0:
                print(
                    f"\nMigration stopped: '{table}' already "
                    f"contains {count} MySQL record(s)."
                )
                print(
                    "No migration records were inserted. "
                    "Check the existing data before continuing."
                )
                return

        target.commit()

        # Insert all tables in a single transaction.
        # Connector starts a transaction automatically on writes.
        for table, columns in TABLE_COLUMNS.items():
            column_sql = ", ".join(
                f"`{column}`" for column in columns
            )
            placeholders = ", ".join(
                ["%s"] * len(columns)
            )

            query = (
                f"INSERT INTO `{table}` ({column_sql}) "
                f"VALUES ({placeholders})"
            )

            rows = source_data[table]

            if rows:
                cursor.executemany(query, rows)
                transaction_started = True

        # Verify the inserted record counts before committing.
        print("\nVerifying MySQL record counts...")

        for table, rows in source_data.items():
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            actual = cursor.fetchone()[0]
            expected = len(rows)

            if actual != expected:
                raise RuntimeError(
                    f"Verification failed for {table}: "
                    f"expected {expected}, found {actual}."
                )

        # Commit only after verification succeeds.
        target.commit()
        transaction_started = False

        print("\nMigration completed successfully!")
        print("Verified MySQL record counts:")

        for table, rows in source_data.items():
            print(f"  {table}: {len(rows)}")

        print("\nThe original SQLite database was not changed.")
        print("Keep your backup until SmartPM is tested.")

    except Exception as exc:
        if target is not None and target.is_connected():
            target.rollback()

        print("\nMigration did not complete.")
        print("Reason:", str(exc))
        print("Any uncommitted changes were rolled back.")

    finally:
        if cursor is not None:
            cursor.close()

        if source is not None:
            source.close()

        if target is not None and target.is_connected():
            target.close()


if __name__ == "__main__":
    main()
