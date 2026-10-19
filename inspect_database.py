
import sqlite3

DB_PATH = "backend/users.db"
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

for table in ["users", "login_attempts"]:
    print(f"\n{'=' * 40}")
    print(f"TABLE: {table}")
    print("=" * 40)

    columns = connection.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    for column in columns:
        print(f"- {column['name']} ({column['type']})")

    count = connection.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print("Total records:", count)

connection.close()
print("\nInspection complete. No data was changed.")
