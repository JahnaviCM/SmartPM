import getpass
import sqlite3

from backend.auth_db import init_db, create_user, get_user_by_username


def main():
    init_db()

    username = input("Enter administrator username: ").strip()

    if get_user_by_username(username):
        print("This username already exists. Choose another username.")
        return

    password = getpass.getpass("Create administrator password (minimum 12 characters): ")
    confirm_password = getpass.getpass("Confirm administrator password: ")

    if password != confirm_password:
        print("Passwords do not match. Please try again.")
        return

    try:
        create_user(
            username=username,
            password=password,
            role="admin",
            approved=True
        )
        print("Administrator account created successfully!")
        print("Keep your username and password private.")

    except (ValueError, sqlite3.IntegrityError) as error:
        print("Could not create administrator account:", error)


if __name__ == "__main__":
    main()