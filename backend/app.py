import os
import time
import hmac
import secrets
import mysql.connector
from functools import wraps
from io import BytesIO
from xml.sax.saxutils import escape

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    abort,
    send_file,
)
from werkzeug.security import check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from backend.auth_db import (
    init_db,
    get_connection,
    get_user_by_username,
    get_user_by_id,
    create_user,
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "random_forest_model.pkl"
)
DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "ai4i2020.csv"
)
STATIC_DIR = os.path.join(BACKEND_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BACKEND_DIR, "templates"),
    static_folder=STATIC_DIR,
)

# =========================================================
# FLASK SECRET KEY CONFIGURATION
# =========================================================

SECRET_KEY = os.environ.get("SECRET_KEY")

if not SECRET_KEY:
    if os.environ.get("RENDER"):
        raise RuntimeError(
            "SECRET_KEY environment variable is missing on Render."
        )

    SECRET_FILE = os.path.join(BACKEND_DIR, ".secret_key")

    if os.path.exists(SECRET_FILE):
        with open(SECRET_FILE, "rb") as secret_file:
            SECRET_KEY = secret_file.read()
    else:
        SECRET_KEY = secrets.token_bytes(32)
        with open(SECRET_FILE, "wb") as secret_file:
            secret_file.write(SECRET_KEY)

app.secret_key = SECRET_KEY

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=1800,
)

PREDICTION_THRESHOLD = 0.40


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

init_db()


def init_security_tables():
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                username VARCHAR(80) PRIMARY KEY,
                failed_count INT NOT NULL DEFAULT 0,
                first_failed DOUBLE NULL,
                locked_until DOUBLE NULL
            )
        """)
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def init_prediction_history_table():
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                created_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
                machine_type VARCHAR(10) NOT NULL,
                air_temperature DOUBLE NOT NULL,
                process_temperature DOUBLE NOT NULL,
                rotational_speed DOUBLE NOT NULL,
                torque DOUBLE NOT NULL,
                tool_wear DOUBLE NOT NULL,
                result VARCHAR(100) NOT NULL,
                failure_probability DOUBLE NOT NULL,
                risk_level VARCHAR(30) NOT NULL,
                threshold DOUBLE NOT NULL,
                recommendation TEXT NOT NULL,
                INDEX idx_prediction_history_user (user_id, id)
            )
        """)
        connection.commit()
    finally:
        cursor.close()
        connection.close()


init_security_tables()
init_prediction_history_table()


# =========================================================
# LOGIN SECURITY
# =========================================================

def login_is_locked(username):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT locked_until
            FROM login_attempts
            WHERE username = %s
        """, (username.lower(),))
        row = cursor.fetchone()
    finally:
        cursor.close()
        connection.close()

    return bool(row and row[0] and row[0] > time.time())


def record_failed_login(username):
    now = time.time()
    username = username.lower()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT failed_count, first_failed
            FROM login_attempts
            WHERE username = %s
        """, (username,))
        row = cursor.fetchone()

        if not row or row[1] is None or now - row[1] > 900:
            count = 1
            first_failed = now
        else:
            count = row[0] + 1
            first_failed = row[1]

        locked_until = now + 900 if count >= 5 else 0

        cursor.execute("""
            INSERT INTO login_attempts
                (username, failed_count, first_failed, locked_until)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                failed_count = VALUES(failed_count),
                first_failed = VALUES(first_failed),
                locked_until = VALUES(locked_until)
        """, (username, count, first_failed, locked_until))

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def clear_failed_logins(username):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            "DELETE FROM login_attempts WHERE username = %s",
            (username.lower(),),
        )
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


# =========================================================
# SESSION AND CSRF SECURITY
# =========================================================

def current_user():
    user_id = session.get("user_id")

    if user_id is None:
        return None

    user = get_user_by_id(user_id)

    if user is None or not user["approved"]:
        session.clear()
        return None

    return user


def login_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash(
                "Please log in to access the dashboard.",
                "error",
            )
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped


def admin_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        user = current_user()

        if user is None:
            return redirect(url_for("login"))

        if user["role"] != "admin":
            abort(403)

        return view_function(*args, **kwargs)

    return wrapped


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        submitted_token = request.form.get("csrf_token", "")
        saved_token = session.get("csrf_token", "")

        if (
            not submitted_token
            or not saved_token
            or not hmac.compare_digest(
                submitted_token,
                saved_token,
            )
        ):
            abort(
                400,
                description="Invalid or missing security token.",
            )

    if session.get("user_id"):
        now = time.time()
        last_activity = session.get("last_activity", now)

        if now - last_activity > 1800:
            session.clear()
            flash(
                "Your session expired. Please log in again.",
                "error",
            )
        else:
            session["last_activity"] = now


@app.context_processor
def add_template_security():
    if not session.get("csrf_token"):
        session["csrf_token"] = secrets.token_urlsafe(32)

    return {
        "csrf_token": session["csrf_token"],
        "logged_in_user": current_user(),
    }


# =========================================================
# LOAD TRAINED RANDOM FOREST MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# LOAD DATASET AND PREPARE MODEL METRICS
# =========================================================

df = pd.read_csv(DATASET_PATH)
df = df.drop(
    columns=["UDI", "Product ID"],
    errors="ignore",
)

df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2,
})

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

X = df[features]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
) * 100
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
) * 100
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
) * 100


def performance_values():
    return {
        "accuracy": round(accuracy, 1),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
    }


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Failure"],
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(
    os.path.join(STATIC_DIR, "confusion_matrix.png")
)
plt.close()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values(
        by="Importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"],
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(
        os.path.join(STATIC_DIR, "feature_importance.png")
    )
    plt.close()


# =========================================================
# WELCOME PAGE
# =========================================================

@app.route("/")
def welcome():
    if current_user() is not None:
        return redirect(url_for("home"))

    return render_template("welcome.html")


# =========================================================
# PROJECT DESCRIPTION
# =========================================================

@app.route("/description")
def description():
    return render_template("description.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user() is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get(
            "username", ""
        ).strip()
        password = request.form.get("password", "")

        if (
            not username
            or len(username) > 80
            or len(password) > 1024
        ):
            return render_template(
                "login.html",
                error="Invalid username or password.",
            ), 400

        if login_is_locked(username):
            return render_template(
                "login.html",
                error=(
                    "Too many failed attempts. "
                    "Try again in 15 minutes."
                ),
            ), 429

        user = get_user_by_username(username)

        if user:
            try:
                valid_password = check_password_hash(
                    user["password_hash"],
                    password,
                )
            except (ValueError, TypeError):
                valid_password = False
        else:
            valid_password = False

        if (
            user is None
            or not valid_password
            or not user["approved"]
        ):
            record_failed_login(username)

            return render_template(
                "login.html",
                error=(
                    "Invalid credentials or account "
                    "not approved."
                ),
            ), 401

        clear_failed_logins(username)

        session.clear()
        session["user_id"] = user["id"]
        session["csrf_token"] = secrets.token_urlsafe(32)
        session["last_activity"] = time.time()
        session.permanent = True

        return redirect(url_for("home"))

    return render_template("login.html")


# =========================================================
# SIGN UP
# =========================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user() is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get(
            "username", ""
        ).strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not username or len(username) > 80:
            return render_template(
                "signup.html",
                error="Username must contain 1–80 characters.",
            ), 400

        if len(password) < 12:
            return render_template(
                "signup.html",
                error=(
                    "Password must contain at least "
                    "12 characters."
                ),
            ), 400

        if len(password) > 1024:
            return render_template(
                "signup.html",
                error="Password is too long.",
            ), 400

        if password != confirm_password:
            return render_template(
                "signup.html",
                error="Passwords do not match.",
            ), 400

        try:
            create_user(
                username=username,
                password=password,
                role="user",
                approved=True,
            )

        except mysql.connector.IntegrityError:
            return render_template(
                "signup.html",
                error="That username is already registered.",
            ), 409

        except ValueError as error:
            return render_template(
                "signup.html",
                error=str(error),
            ), 400

        flash(
            "Registration successful! Your account is "
            "approved. You can now log in.",
            "success",
        )

        return redirect(url_for("login"))

    return render_template("signup.html")


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required
def home():
    return render_template(
        "index.html",
        **performance_values(),
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin")
@admin_required
def admin_dashboard():
    user = current_user()

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT id, username, role, approved
            FROM users
            ORDER BY id ASC
        """)
        users = cursor.fetchall()

    finally:
        cursor.close()
        connection.close()

    return render_template(
        "admin.html",
        users=users,
        admin_user=user,
    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/history")
@login_required
def history():
    user = current_user()

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT *
            FROM prediction_history
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT 100
        """, (user["id"],))

        records = cursor.fetchall()

    finally:
        cursor.close()
        connection.close()

    return render_template(
        "history.html",
        records=records,
    )


# =========================================================
# MACHINE PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    prediction_id = None

    try:
        machine_type = request.form.get(
            "machine_type", ""
        ).strip().upper()

        type_values = {
            "L": 0,
            "M": 1,
            "H": 2,
        }

        if machine_type not in type_values:
            raise ValueError(
                "Please select a valid machine type."
            )

        air_temperature = float(
            request.form.get("air_temperature", "")
        )
        process_temperature = float(
            request.form.get("process_temperature", "")
        )
        rotational_speed = float(
            request.form.get("rotational_speed", "")
        )
        torque = float(
            request.form.get("torque", "")
        )
        tool_wear = float(
            request.form.get("tool_wear", "")
        )

        values = [
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        ]

        if not all(pd.notna(value) for value in values):
            raise ValueError(
                "All measurements must be valid numbers."
            )

        if not 250 <= air_temperature <= 350:
            raise ValueError(
                "Air temperature must be between 250 and 350 K."
            )

        if not 250 <= process_temperature <= 400:
            raise ValueError(
                "Process temperature must be between 250 and 400 K."
            )

        if not 0 < rotational_speed <= 10000:
            raise ValueError(
                "Rotational speed must be between 1 and 10000 RPM."
            )

        if not 0 <= torque <= 1000:
            raise ValueError(
                "Torque must be between 0 and 1000 Nm."
            )

        if not 0 <= tool_wear <= 1000:
            raise ValueError(
                "Tool wear must be between 0 and 1000 minutes."
            )

        input_data = pd.DataFrame(
            [[
                type_values[machine_type],
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
            ]],
            columns=features,
        )

        # Obtain probability for the failure class.
        classes = list(model.classes_)

        if 1 not in classes:
            raise RuntimeError(
                "The loaded model does not contain failure class 1."
            )

        failure_class_index = classes.index(1)

        probabilities = model.predict_proba(input_data)[0]
        failure_probability_raw = float(
            probabilities[failure_class_index]
        )

        failure_percentage = round(
            failure_probability_raw * 100,
            2,
        )

        prediction = int(
            failure_probability_raw >= PREDICTION_THRESHOLD
        )

        if failure_probability_raw < 0.20:
            risk_level = "Low"
            risk_status = "low"

        elif failure_probability_raw < PREDICTION_THRESHOLD:
            risk_level = "Medium"
            risk_status = "medium"

        else:
            risk_level = "High"
            risk_status = "high"

        if prediction == 0:
            result = "MACHINE NORMAL"

            message = (
                "The model predicts normal operation. "
                "Continue routine monitoring and follow "
                "maintenance guidance."
            )

            status = "normal"

            recommendation = (
                "Continue routine monitoring and follow "
                "the scheduled maintenance plan."
            )

        else:
            result = "MACHINE FAILURE"

            message = (
                "The model predicts a potential failure. "
                "Arrange an inspection and follow your "
                "maintenance procedures."
            )

            status = "failure"

            recommendation = (
                "Inspect the machine and review its "
                "maintenance requirements."
            )

        user = current_user()

        if user is None:
            flash(
                "Your session expired. Please log in again.",
                "error",
            )
            return redirect(url_for("login"))

        # Save the prediction in MySQL.
        connection = get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("""
                INSERT INTO prediction_history (
                    user_id,
                    created_at,
                    machine_type,
                    air_temperature,
                    process_temperature,
                    rotational_speed,
                    torque,
                    tool_wear,
                    result,
                    failure_probability,
                    risk_level,
                    threshold,
                    recommendation
                )
                VALUES (
                    %s,
                    NOW(),
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
            """, (
                user["id"],
                machine_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                result,
                failure_percentage,
                risk_level,
                PREDICTION_THRESHOLD,
                recommendation,
            ))

            # Capture the new record ID immediately.
            prediction_id = cursor.lastrowid
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            cursor.close()
            connection.close()

        # Render the dashboard with the prediction values.
        return render_template(
            "index.html",
            prediction_id=prediction_id,
            result=result,
            message=message,
            status=status,
            failure_probability=failure_percentage,
            risk_level=risk_level,
            risk_status=risk_status,
            threshold=PREDICTION_THRESHOLD,
            recommendation=recommendation,
            machine_type=machine_type,
            air_temperature=air_temperature,
            process_temperature=process_temperature,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            **performance_values(),
        )

    except (ValueError, KeyError, TypeError) as error:
        # Show validation errors on the dashboard instead
        # of leaving the user without a visible message.
        return render_template(
            "index.html",
            result="INVALID INPUT",
            message=str(error) or (
                "Please enter valid values for all machine measurements."
            ),
            status="failure",
            risk_status="high",
            **performance_values(),
        ), 400


# =========================================================
# DOWNLOAD PREDICTION REPORT AS PDF
# =========================================================

@app.route(
    "/prediction/<int:prediction_id>/report.pdf"
)
@login_required
def download_prediction_report(prediction_id):
    user = current_user()

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT *
            FROM prediction_history
            WHERE id = %s AND user_id = %s
        """, (
            prediction_id,
            user["id"],
        ))

        record = cursor.fetchone()

    finally:
        cursor.close()
        connection.close()

    # Users can only download their own reports.
    if record is None:
        abort(404)

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"SmartPM Prediction Report {prediction_id}",
        author="SmartPM Predictive Maintenance System",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SmartPMTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#30451b"),
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "SmartPMSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#59636b"),
        spaceAfter=16,
    )

    story = [
        Paragraph("SmartPM", title_style),
        Paragraph(
            "SMART PREDICTIVE MAINTENANCE — PREDICTION REPORT",
            subtitle_style,
        ),
        Paragraph(
            f"<b>Report ID:</b> SPM-{int(record['id']):06d}",
            styles["Normal"],
        ),
        Paragraph(
            f"<b>Prediction date:</b> "
            f"{escape(str(record.get('created_at') or 'Not recorded'))}",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Prediction Summary", styles["Heading2"]),
    ]

    result_text = escape(
        str(record.get("result", "Not available"))
    )

    summary_data = [
        ["Predicted status", result_text],
        [
            "Failure probability",
            f"{float(record.get('failure_probability') or 0):.2f}%",
        ],
        [
            "Risk level",
            escape(str(record.get("risk_level", "Not available"))),
        ],
        [
            "Prediction threshold",
            f"{float(record.get('threshold') or 0) * 100:.0f}%",
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[52 * mm, 105 * mm],
    )

    summary_table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (0, -1),
            colors.HexColor("#edf2e6"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, -1),
            colors.HexColor("#172019"),
        ),
        (
            "FONTNAME",
            (0, 0),
            (0, -1),
            "Helvetica-Bold",
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.HexColor("#cbd5c4"),
        ),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    story.extend([
        summary_table,
        Spacer(1, 16),
        Paragraph(
            "Machine Sensor Readings",
            styles["Heading2"],
        ),
    ])

    sensor_data = [
        ["Parameter", "Recorded value", "Unit"],
        [
            "Machine type",
            escape(str(record.get("machine_type", "—"))),
            "L / M / H",
        ],
        [
            "Air temperature",
            f"{float(record.get('air_temperature') or 0):.2f}",
            "K",
        ],
        [
            "Process temperature",
            f"{float(record.get('process_temperature') or 0):.2f}",
            "K",
        ],
        [
            "Rotational speed",
            f"{float(record.get('rotational_speed') or 0):.2f}",
            "RPM",
        ],
        [
            "Torque",
            f"{float(record.get('torque') or 0):.2f}",
            "Nm",
        ],
        [
            "Tool wear",
            f"{float(record.get('tool_wear') or 0):.2f}",
            "minutes",
        ],
    ]

    sensor_table = Table(
        sensor_data,
        colWidths=[65 * mm, 48 * mm, 44 * mm],
        repeatRows=1,
    )

    sensor_table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#26343c"),
        ),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#f5f7f3")],
        ),
        (
            "TEXTCOLOR",
            (0, 1),
            (-1, -1),
            colors.HexColor("#172019"),
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.HexColor("#cbd5c4"),
        ),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.extend([
        sensor_table,
        Spacer(1, 16),
        Paragraph(
            "Maintenance Recommendation",
            styles["Heading2"],
        ),
    ])

    recommendation = escape(
        str(
            record.get(
                "recommendation",
                "Follow the site's maintenance procedures.",
            )
        )
    )

    story.append(
        Paragraph(recommendation, styles["BodyText"])
    )

    story.extend([
        Spacer(1, 18),
        Paragraph(
            "Disclaimer: This report contains a machine-learning "
            "estimate based on the values entered. It is not a "
            "guarantee of machine failure or safety. Follow "
            "qualified maintenance procedures.",
            styles["Italic"],
        ),
        Spacer(1, 10),
        Paragraph(
            "Generated by SmartPM — Smart Predictive Maintenance System",
            subtitle_style,
        ),
    ])

    document.build(story)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=(
            f"SmartPM_Prediction_Report_{prediction_id}.pdf"
        ),
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()

    flash(
        "You have logged out successfully.",
        "success",
    )

    return redirect(url_for("login"))


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(403)
def forbidden(error):
    return (
        "Access denied. You are not authorized to view this page.",
        403,
    )


@app.errorhandler(400)
def bad_request(error):
    return (
        "Request rejected. Refresh the page and try again.",
        400,
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )