import sqlite3

connection = sqlite3.connect("backend/users.db")

query = "SELECT name FROM sqlite_master WHERE type = ?"
tables = connection.execute(query, ("table",)).fetchall()

print("\nTables in your existing database:")
for table in tables:
    print("-", table[0])

connection.close()