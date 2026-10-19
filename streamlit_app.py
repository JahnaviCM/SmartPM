
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# SMARTPM | PREMIUM INDUSTRIAL AI DASHBOARD
# =========================================================

st.set_page_config(
    page_title="SmartPM | Industrial Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_FILES = [
    ROOT / "model" / "improved_random_forest_model.pkl",
    ROOT / "model" / "random_forest_model.pkl",
    ROOT / "backend" / "model" / "improved_random_forest_model.pkl",
    ROOT / "backend" / "model" / "random_forest_model.pkl",
    ROOT / "improved_random_forest_model.pkl",
    ROOT / "random_forest_model.pkl",
]
DATA_FILES = [
    ROOT / "dataset" / "ai4i2020.csv",
    ROOT / "dataset" / "ai4i2020_train.csv",
    ROOT / "ai4i2020.csv",
]

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_MAP = {"L": 0, "M": 1, "H": 2}

BG = "#090D12"
PANEL = "#111A22"
PANEL2 = "#16222B"
BORDER = "#283842"
GREEN = "#C9F269"
MINT = "#75E0B4"
RED = "#FF7D89"
AMBER = "#F5C56A"
MUTED = "#9EADB7"
WHITE = "#F5F7F2"


# =========================================================
# VISUAL DESIGN
# =========================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {{
        --bg: {BG};
        --panel: {PANEL};
        --border: {BORDER};
        --green: {GREEN};
        --muted: {MUTED};
    }}

    .stApp {{
        background:
            radial-gradient(ellipse at 85% 0%, #18281e 0%, transparent 34%),
            radial-gradient(ellipse at 0% 40%, #101923 0%, transparent 35%),
            {BG};
        color: {WHITE};
        font-family: 'DM Sans', sans-serif;
    }}

    [data-testid="stHeader"] {{
        background: rgba(9,13,18,.9);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0D151C, #0A1015);
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        font-family: 'DM Sans', sans-serif;
    }}

    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {WHITE} !important;
        letter-spacing: -0.6px;
    }}

    h1 {{ font-size: 2.2rem !important; }}
    h2 {{ font-size: 1.6rem !important; }}
    p {{ line-height: 1.65; }}

    .brand-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 7px 0 18px;
    }}

    .brand-icon {{
        width: 43px;
        height: 43px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 13px;
        color: #10170D;
        font-size: 23px;
        background: linear-gradient(135deg, {GREEN}, #92D65A);
        box-shadow: 0 0 25px #C9F26925;
    }}

    .brand-name {{
        color: {WHITE};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 23px;
        font-weight: 700;
    }}

    .brand-name span {{ color: {GREEN}; }}

    .brand-caption {{
        color: {MUTED};
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #17261F 0%, #121D24 52%, #10171F 100%);
        border: 1px solid #3A5040;
        border-radius: 23px;
        padding: 33px 34px;
        margin: 4px 0 27px;
        box-shadow: 0 18px 60px #00000024;
    }}

    .hero:after {{
        content: "";
        position: absolute;
        right: -65px;
        top: -130px;
        width: 320px;
        height: 320px;
        border: 1px solid #C9F26920;
        border-radius: 50%;
        box-shadow: 0 0 0 30px #C9F26908, 0 0 0 65px #C9F26905;
        pointer-events: none;
    }}

    .eyebrow {{
        color: {GREEN};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2.4px;
        text-transform: uppercase;
        margin-bottom: 13px;
    }}

    .hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(28px, 3.1vw, 43px);
        line-height: 1.13;
        letter-spacing: -1.6px;
        font-weight: 700;
        color: {WHITE};
        margin-bottom: 14px;
    }}

    .hero-title span {{ color: {GREEN}; }}

    .hero-description {{
        max-width: 680px;
        color: #B5C1C5;
        font-size: 13px;
        line-height: 1.9;
    }}

    .live-pill {{
        display: inline-block;
        color: {GREEN};
        background: #C9F26912;
        border: 1px solid #C9F26935;
        border-radius: 30px;
        padding: 7px 12px;
        margin-top: 16px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.3px;
    }}

    .section-label {{
        color: {GREEN};
        font-size: 10px;
        letter-spacing: 2.3px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 7px;
    }}

    .section-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 23px;
        font-weight: 700;
        color: {WHITE};
        letter-spacing: -.7px;
    }}

    .section-desc {{
        color: {MUTED};
        font-size: 12px;
        margin: 5px 0 20px;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, #152029, #10171E);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 19px 19px;
        box-shadow: 0 8px 25px #00000015;
        transition: border-color .2s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: #617B53;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 11px;
    }}

    div[data-testid="stMetricValue"] {{
        color: {GREEN};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: 11px;
    }}

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {{
        color: #11180D;
        background: linear-gradient(110deg, {GREEN}, #A9E96D);
        border: 0;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
        transition: .2s ease;
    }}

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {{
        color: #11180D;
        background: #DFFF91;
        box-shadow: 0 5px 24px #C9F26924;
        border: 0;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: #141F28;
        border-color: {BORDER};
        border-radius: 9px;
    }}

    div[data-testid="stExpander"] {{
        background: #111A22;
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    .info-card {{
        background: linear-gradient(145deg, #152029, #10171E);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 22px;
        min-height: 142px;
    }}

    .info-card .number {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: {GREEN};
        margin: 10px 0;
    }}

    .info-card .label {{
        color: {MUTED};
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 1.5px;
    }}

    .info-card .description {{
        color: #B7C1C5;
        font-size: 12px;
    }}

    .result-card {{
        border: 1px solid {BORDER};
        background: linear-gradient(140deg, #17231F, #111A22);
        border-radius: 17px;
        padding: 22px;
    }}

    .result-caption {{
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 1.7px;
        font-size: 10px;
    }}

    .result-number {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 31px;
        font-weight: 700;
        margin-top: 9px;
    }}

    .footer {{
        text-align: center;
        color: #73818A;
        font-size: 10px;
        letter-spacing: 1.3px;
        border-top: 1px solid {BORDER};
        padding: 24px 0 8px;
        margin-top: 35px;
    }}

    hr {{ border-color: {BORDER}; }}

    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# RESOURCE LOADING
# =========================================================

@st.cache_resource
def load_model():
    for path in MODEL_FILES:
        if path.exists():
            return joblib.load(path), path
    return None, None


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


try:
    model, model_path = load_model()
except Exception as exc:
    model, model_path = None, None
    model_error = str(exc)
else:
    model_error = None

data = None
data_path = None

for path in DATA_FILES:
    if path.exists():
        try:
            data = read_csv(str(path))
            data_path = path
            break
        except Exception:
            continue


# =========================================================
# DATA PREPARATION
# =========================================================

def locate_column(frame, choices):
    for choice in choices:
        if choice in frame.columns:
            return choice

    lookup = {str(col).lower(): col for col in frame.columns}
    for choice in choices:
        if choice.lower() in lookup:
            return lookup[choice.lower()]

    return None


def normalize_dataset(frame):
    frame = frame.copy()

    aliases = {
        "Type": ["Type", "type"],
        "Air temperature [K]": [
            "Air temperature [K]", "Air temperature"
        ],
        "Process temperature [K]": [
            "Process temperature [K]", "Process temperature"
        ],
        "Rotational speed [rpm]": [
            "Rotational speed [rpm]", "Rotational speed"
        ],
        "Torque [Nm]": ["Torque [Nm]", "Torque"],
        "Tool wear [min]": ["Tool wear [min]", "Tool wear"],
        "Machine failure": [
            "Machine failure", "machine_failure", "Failure"
        ],
    }

    for destination, names in aliases.items():
        source = locate_column(frame, names)
        if source is not None and source != destination:
            frame[destination] = frame[source]

    if "Type" in frame.columns:
        frame["Type"] = (
            frame["Type"].astype(str).str.strip().str.upper()
        )

    return frame


if data is not None:
    data = normalize_dataset(data)


# =========================================================
# CHART HELPERS
# =========================================================

def section(label, title, description):
    st.markdown(
        f"""
        <div class="section-label">{label}</div>
        <div class="section-title">{title}</div>
        <div class="section-desc">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=WHITE, family="DM Sans", size=11),
        margin=dict(l=22, r=20, t=48, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
    )
    fig.update_yaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
    )
    return fig


def predict_values(input_frame):
    prediction = int(model.predict(input_frame)[0])
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    return prediction, probability


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">⚙</div>
            <div>
                <div class="brand-name">Smart<span>PM</span></div>
                <div class="brand-caption">Industrial intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Failure Prediction",
            "Visual Analytics",
            "Model Insights",
            "About SmartPM",
        ],
    )

    st.divider()
    st.caption("SYSTEM STATUS")

    if model is not None:
        st.success("Random Forest loaded")
    else:
        st.error("Model unavailable")

    if data is not None:
        st.success("AI4I dataset loaded")
    else:
        st.warning("Dataset not found")

    st.divider()
    st.markdown("**TECH STACK**")
    st.caption("Python · Random Forest")
    st.caption("Streamlit · Plotly")
    st.caption("AI4I 2020 Dataset")

    st.divider()
    st.caption("SmartPM · Predict before failure")


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMARTPM / PREDICTIVE INTELLIGENCE</div>
        <div class="hero-title">
            Machine intelligence.<br>
            <span>Before failure happens.</span>
        </div>
        <div class="hero-description">
            Monitor operating conditions, analyze machine behavior,
            and estimate failure risk through machine learning.
            A unified workspace for predictive maintenance analytics.
        </div>
        <div class="live-pill">✦ &nbsp; INDUSTRIAL ANALYTICS WORKSPACE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(f"Could not load model: {model_error}")

if model is None:
    st.error("The saved model file was not found.")
    st.write("Check that your model exists in the project's model folder.")
    st.code(
        "model/random_forest_model.pkl\n"
        "model/improved_random_forest_model.pkl"
    )
    st.stop()


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    section(
        "01 / Command center",
        "Operational overview",
        "A consolidated view of the loaded model and available dataset.",
    )

    records = len(data) if data is not None else 0
    failures = 0
    failure_rate = None

    if data is not None and "Machine failure" in data.columns:
        labels = pd.to_numeric(
            data["Machine failure"], errors="coerce"
        ).dropna()
        failures = int((labels == 1).sum())
        if len(labels):
            failure_rate = float((labels == 1).mean() * 100)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("DATASET RECORDS", f"{records:,}")

    with c2:
        st.metric(
            "RECORDED FAILURES",
            f"{failures:,}" if data is not None else "N/A",
        )

    with c3:
        st.metric(
            "OBSERVED FAILURE RATE",
            f"{failure_rate:.2f}%"
            if failure_rate is not None else "N/A",
        )

    with c4:
        st.metric("ML ALGORITHM", "Random Forest")

    st.write("")

    left, right = st.columns([1.15, 0.85])

    with left:
        section(
            "DATASET INTELLIGENCE",
            "Failure distribution",
            "Actual labels in your dataset, if available.",
        )

        if data is not None and "Machine failure" in data.columns:
            labels = pd.to_numeric(
                data["Machine failure"], errors="coerce"
            ).dropna()

            counts = labels.map({
                0: "No failure",
                1: "Failure",
            }).value_counts().rename_axis("Condition").reset_index(
                name="Records"
            )

            fig = px.pie(
                counts,
                names="Condition",
                values="Records",
                hole=0.72,
                color="Condition",
                color_discrete_map={
                    "No failure": GREEN,
                    "Failure": RED,
                },
            )

            fig.update_traces(
                textinfo="label+percent",
                textposition="outside",
                marker=dict(line=dict(color=PANEL, width=4)),
            )

            fig.add_annotation(
                text=f"<b>{len(labels):,}</b><br><sup>records</sup>",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=22, color=WHITE),
            )

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
            )
        else:
            st.info(
                "Add dataset/ai4i2020.csv to enable dataset analytics."
            )

    with right:
        section(
            "SYSTEM HEALTH",
            "Resource status",
            "Files currently detected by SmartPM.",
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="label">Machine learning engine</div>
                <div class="number">Random Forest</div>
                <div class="description">Saved model loaded successfully.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="info-card">
                <div class="label">Analytics interface</div>
                <div class="number">Streamlit + Plotly</div>
                <div class="description">Interactive visualizations and model predictions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if model_path:
            st.caption(f"Model: {model_path.name}")

        if data_path:
            st.caption(f"Dataset: {data_path.name}")

    st.divider()

    section(
        "WORKFLOW",
        "From readings to risk assessment",
        "Three stages in the current predictive workflow.",
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="info-card">
                <div class="label">01 / INPUT</div>
                <div class="number">Collect</div>
                <div class="description">
                    Enter six machine operating measurements.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="info-card">
                <div class="label">02 / ANALYZE</div>
                <div class="number">Predict</div>
                <div class="description">
                    Process inputs with the trained Random Forest.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            """
            <div class="info-card">
                <div class="label">03 / REVIEW</div>
                <div class="number">Act</div>
                <div class="description">
                    Review estimated risk and consider inspection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FAILURE PREDICTION
# =========================================================

elif page == "Failure Prediction":

    section(
        "02 / Prediction engine",
        "Machine risk assessment",
        "Configure operating conditions and analyze them with your model.",
    )

    st.info(
        "This dashboard uses manual inputs. It is not connected to live "
        "industrial sensors."
    )

    with st.form("smartpm_prediction"):

        left, right = st.columns(2)

        with left:
            machine_type = st.selectbox(
                "Machine type",
                ["L", "M", "H"],
                format_func=lambda value: {
                    "L": "L — Low variant",
                    "M": "M — Medium variant",
                    "H": "H — High variant",
                }[value],
            )

            air = st.number_input(
                "Air temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=298.1,
                step=0.1,
            )

            process = st.number_input(
                "Process temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=308.6,
                step=0.1,
            )

        with right:
            speed = st.number_input(
                "Rotational speed (RPM)",
                min_value=1,
                max_value=10000,
                value=1500,
                step=10,
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=1000.0,
                value=40.0,
                step=0.5,
            )

            wear = st.number_input(
                "Tool wear (minutes)",
                min_value=0,
                max_value=1000,
                value=100,
                step=5,
            )

        threshold = st.slider(
            "Risk classification threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Adjusts the risk label, not the model's underlying probability.",
        )

        submitted = st.form_submit_button(
            "⚡  ANALYZE MACHINE",
            use_container_width=True,
        )

    if submitted:
        values = [
            TYPE_MAP[machine_type],
            air,
            process,
            speed,
            torque,
            wear,
        ]

        input_frame = pd.DataFrame([values], columns=FEATURES)

        try:
            prediction, probability = predict_values(input_frame)

            st.divider()
            section(
                "ANALYSIS COMPLETE",
                "Prediction report",
                "Results from the saved Random Forest model.",
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "MODEL CLASSIFICATION",
                    "Failure" if prediction == 1 else "No failure",
                )

            with m2:
                st.metric(
                    "FAILURE PROBABILITY",
                    f"{probability * 100:.2f}%"
                    if probability is not None else "Unavailable",
                )

            with m3:
                st.metric("DECISION THRESHOLD", f"{threshold * 100:.0f}%")

            if probability is not None:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={
                            "suffix": "%",
                            "font": {"size": 38, "color": WHITE},
                        },
                        title={
                            "text": "ESTIMATED FAILURE PROBABILITY",
                            "font": {"size": 12, "color": MUTED},
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": MUTED,
                            },
                            "bar": {"color": GREEN, "thickness": 0.25},
                            "bgcolor": PANEL,
                            "bordercolor": BORDER,
                            "steps": [
                                {
                                    "range": [0, threshold * 100],
                                    "color": "#20342A",
                                },
                                {
                                    "range": [threshold * 100, 100],
                                    "color": "#40262C",
                                },
                            ],
                            "threshold": {
                                "line": {"color": RED, "width": 4},
                                "thickness": 0.8,
                                "value": threshold * 100,
                            },
                        },
                    )
                )

                gauge.update_layout(
                    height=310,
                    paper_bgcolor=PANEL,
                    font={"color": WHITE},
                    margin={"l": 25, "r": 25, "t": 55, "b": 20},
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True,
                )

                if probability >= threshold:
                    st.error(
                        "HIGHER PREDICTED RISK — Consider reviewing the "
                        "operating measurements and inspecting the machine."
                    )
                else:
                    st.success(
                        "LOWER PREDICTED RISK — The estimated probability "
                        "is below the selected threshold."
                    )

            elif prediction == 1:
                st.error("The model predicts machine failure.")
            else:
                st.success("The model predicts no machine failure.")

            with st.expander("View submitted machine measurements"):
                st.dataframe(
                    input_frame,
                    hide_index=True,
                    use_container_width=True,
                )

            st.caption(
                "Predictions are estimates, not guarantees of machine "
                "condition. Validate thresholds using held-out test data."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the saved model uses the same six features, "
                "encoding, and feature order."
            )


# =========================================================
# VISUAL ANALYTICS
# =========================================================

elif page == "Visual Analytics":

    section(
        "03 / Analytics studio",
        "Explore machine behavior",
        "Interactive charts based on the actual dataset when available.",
    )

    if data is None:
        st.warning(
            "Dataset not found. Add dataset/ai4i2020.csv to enable "
            "the dataset visualizations."
        )

    elif "Machine failure" not in data.columns:
        st.warning("The dataset is missing the machine-failure column.")

    else:
        frame = data.copy()

        frame["Machine failure"] = pd.to_numeric(
            frame["Machine failure"], errors="coerce"
        )

        frame["Failure status"] = frame["Machine failure"].map({
            0: "No failure",
            1: "Failure",
        })

        sensors = [name for name in FEATURES if name in frame.columns]

        if not sensors:
            st.warning("No expected sensor columns were detected.")
        else:
            section(
                "SENSOR EXPLORER",
                "Measurement distribution",
                "Compare measurements across failure and non-failure records.",
            )

            sensor = st.selectbox("Select a sensor", sensors)

            fig = px.histogram(
                frame,
                x=sensor,
                color="Failure status",
                nbins=35,
                barmode="overlay",
                opacity=0.8,
                color_discrete_map={
                    "No failure": GREEN,
                    "Failure": RED,
                },
            )

            st.plotly_chart(
                style_chart(fig, 360),
                use_container_width=True,
            )

            left, right = st.columns(2)

            with left:
                section(
                    "SENSOR RELATIONSHIPS",
                    "Torque vs. speed",
                    "Explore two important operating measurements.",
                )

                if (
                    "Torque [Nm]" in frame.columns
                    and "Rotational speed [rpm]" in frame.columns
                ):
                    sample = frame.dropna(
                        subset=[
                            "Torque [Nm]",
                            "Rotational speed [rpm]",
                            "Failure status",
                        ]
                    )

                    if len(sample) > 2500:
                        sample = sample.sample(
                            n=2500, random_state=42
                        )

                    fig = px.scatter(
                        sample,
                        x="Rotational speed [rpm]",
                        y="Torque [Nm]",
                        color="Failure status",
                        opacity=0.7,
                        color_discrete_map={
                            "No failure": GREEN,
                            "Failure": RED,
                        },
                    )

                    st.plotly_chart(
                        style_chart(fig, 350),
                        use_container_width=True,
                    )
                else:
                    st.info("Required sensor columns are unavailable.")

            with right:
                section(
                    "PRODUCT TYPES",
                    "Failures by machine type",
                    "Compare recorded failure counts across L, M and H.",
                )

                if "Type" in frame.columns:
                    counts = (
                        frame.dropna(subset=["Failure status"])
                        .groupby(["Type", "Failure status"])
                        .size()
                        .reset_index(name="Records")
                    )

                    fig = px.bar(
                        counts,
                        x="Type",
                        y="Records",
                        color="Failure status",
                        barmode="group",
                        color_discrete_map={
                            "No failure": GREEN,
                            "Failure": RED,
                        },
                        category_orders={"Type": ["L", "M", "H"]},
                    )

                    st.plotly_chart(
                        style_chart(fig, 350),
                        use_container_width=True,
                    )
                else:
                    st.info("Machine type information is unavailable.")

            with st.expander("Inspect dataset records"):
                st.dataframe(
                    frame.head(100),
                    hide_index=True,
                    use_container_width=True,
                )


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    section(
        "04 / Model intelligence",
        "Understand the model",
        "Inspect supported feature importance and saved evaluation results.",
    )

    estimator = model

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        if steps:
            estimator = steps[-1]

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_)

        if len(importance) == len(FEATURES):
            importance_frame = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": importance,
            }).sort_values("Importance", ascending=True)

            fig = px.bar(
                importance_frame,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#27372D", "#76A94B", GREEN],
            )

            st.plotly_chart(
                style_chart(fig, 410),
                use_container_width=True,
            )

            st.caption(
                "Feature importance reflects how the fitted model used "
                "features. It does not establish causation."
            )
        else:
            st.warning(
                "Feature importance does not match the six expected inputs."
            )
    else:
        st.info(
            "Feature importance is unavailable for this saved estimator."
        )

    st.divider()

    section(
        "MODEL EVALUATION",
        "Available performance results",
        "Shows existing CSV results; no accuracy values are fabricated.",
    )

    result_files = [
        ROOT / "model" / "model_comparison.csv",
        ROOT / "model" / "model_comparison_results.csv",
        ROOT / "model" / "improved_model_results.csv",
        ROOT / "model" / "threshold_results.csv",
        ROOT / "model" / "threshold_comparison.csv",
        ROOT / "model" / "best_comparison_results.csv",
    ]

    found = False

    for result_file in result_files:
        if result_file.exists():
            try:
                result = pd.read_csv(result_file)
                if not result.empty:
                    st.markdown(f"**Source:** `{result_file.name}`")
                    st.dataframe(
                        result,
                        hide_index=True,
                        use_container_width=True,
                    )
                    found = True
                    break
            except Exception:
                continue

    if not found:
        st.info(
            "No recognized evaluation CSV was found. We can connect "
            "your actual model-evaluation file separately."
        )


# =========================================================
# ABOUT
# =========================================================

elif page == "About SmartPM":

    section(
        "05 / Project profile",
        "Smart Predictive Maintenance",
        "Machine learning for data-driven maintenance decisions.",
    )

    st.markdown(
        """
        ### Project objective

        Estimate machine failure risk from operating measurements using
        a trained Random Forest classifier.

        ### Technology stack

        - **Machine learning:** Random Forest
        - **Dataset:** AI4I 2020 Predictive Maintenance Dataset
        - **Training:** Google Colab
        - **Dashboard:** Streamlit
        - **Visualizations:** Plotly
        - **Existing web application:** Flask, HTML, CSS and JavaScript
        - **Planned database integration:** MySQL

        ### Model inputs

        1. Machine type
        2. Air temperature
        3. Process temperature
        4. Rotational speed
        5. Torque
        6. Tool wear

        ### Current limitations

        - Inputs are entered manually.
        - Live IoT sensor integration is not implemented.
        - Prediction history is not yet stored in MySQL from this dashboard.
        - Performance claims must be based on actual test-set evaluation.
        """
    )

    st.info(
        "This is a separate Streamlit dashboard. Your existing Flask "
        "application remains unchanged."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        SMARTPM &nbsp; · &nbsp; INDUSTRIAL INTELLIGENCE<br><br>
        PREDICTIVE MAINTENANCE &nbsp; / &nbsp;
        MACHINE LEARNING &nbsp; / &nbsp; ANALYTICS
    </div>
    """,
    unsafe_allow_html=True,
)
