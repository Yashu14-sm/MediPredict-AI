
from pathlib import Path
import html
import textwrap

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "datasets" / "processed"

DISEASES = {
    "Diabetes": {
        "model": MODEL_DIR / "diabetes_best_model.joblib",
        "data": DATA_DIR / "diabetes_cleaned.csv",
        "icon": "🩸",
        "description": "Pima Indians Diabetes Dataset",
        "positive": "Diabetes-positive pattern detected",
        "negative": "Diabetes-negative pattern detected"
    },

    "Heart Disease": {
        "model": MODEL_DIR / "heart_disease_best_model.joblib",
        "data": DATA_DIR / "heart_disease_cleaned.csv",
        "icon": "❤️",
        "description": "UCI Heart Disease Dataset",
        "positive": "Heart-disease-positive pattern detected",
        "negative": "Heart-disease-negative pattern detected"
    },

    "Breast Cancer": {
        "model": MODEL_DIR / "breast_cancer_best_model.joblib",
        "data": DATA_DIR / "breast_cancer_cleaned.csv",
        "icon": "🔬",
        "description": "Breast Cancer Wisconsin Diagnostic Dataset",
        "positive": "Malignant-class pattern detected",
        "negative": "Benign-class pattern detected"
    }
}


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. HTML RENDERING FUNCTION
# IMPORTANT: Use st.html, NOT st.markdown.
# ============================================================

def render_html(content):

    st.html(
        textwrap.dedent(content).strip()
    )


# ============================================================
# 4. COMPLETE CUSTOM CSS
# ============================================================

render_html("""
<style>

/* ================= GLOBAL ================= */

.stApp {
    background: #111827 !important;
    color: #F8FAFC !important;
}

.main {
    background: #111827 !important;
}

.main .block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 3rem 2.5rem;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #F8FAFC !important;
}

.stApp hr {
    border-color: #334155 !important;
    margin: 30px 0;
}

/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        120deg,
        #102A43,
        #087E8B,
        #176B87
    );

    padding: 42px;
    border-radius: 22px;
    margin-bottom: 30px;

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin: 0 0 14px 0 !important;
}

.hero p {
    color: #E0F2FE !important;
    font-size: 17px !important;
    line-height: 1.8 !important;
    max-width: 850px;
    margin: 0 !important;
}

/* ================= OVERVIEW CARDS ================= */

.info-card {
    background: #FFFFFF !important;
    padding: 28px;
    border-radius: 17px;
    min-height: 145px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 7px 20px rgba(0,0,0,0.16);
}

.info-card h2 {
    color: #102A43 !important;
    font-size: 39px !important;
    font-weight: 800 !important;
    margin: 0 0 18px 0 !important;
}

.info-card p {
    color: #475569 !important;
    font-size: 17px !important;
    font-weight: 650 !important;
    margin: 0 !important;
}

/* ================= SECTION HEADINGS ================= */

.section-heading {
    color: #F8FAFC !important;
    font-size: 30px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 12px;
}

.section-description {
    color: #AEBCCB !important;
    font-size: 15px;
    margin-bottom: 22px;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #102A43 !important;
    border-right: 1px solid #29445E;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #36516D !important;
}

/* Disease radio options */

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #173B58 !important;
    border: 1px solid #315B78 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    margin-bottom: 10px !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #205273 !important;
    border-color: #4DA7B5 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #FFFFFF !important;
    font-weight: 650 !important;
}

/* ================= PATIENT FORM ================= */

div[data-testid="stForm"] {
    background: #FFFFFF !important;
    padding: 30px !important;
    border-radius: 19px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}

div[data-testid="stForm"] label,
div[data-testid="stForm"] label p,
div[data-testid="stForm"] [data-testid="stWidgetLabel"],
div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
    color: #263449 !important;
    -webkit-text-fill-color: #263449 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

div[data-testid="stForm"] input {
    background: #F8FAFC !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

div[data-testid="stForm"] [data-testid="stNumberInput"] button {
    background: #E2E8F0 !important;
    color: #102A43 !important;
    border: 1px solid #CBD5E1 !important;
}

div[data-testid="stForm"] [data-testid="stNumberInput"] button:hover {
    background: #CBD5E1 !important;
}

div[data-testid="stForm"] input:focus {
    border: 1px solid #087E8B !important;
    box-shadow: 0 0 0 1px #087E8B !important;
}

/* ================= PREDICT BUTTON ================= */

div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background: #087E8B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 11px !important;
    min-height: 54px;
    font-size: 17px !important;
    font-weight: 750 !important;
}

div[data-testid="stForm"] button[kind="primaryFormSubmit"] * {
    color: #FFFFFF !important;
}

div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
    background: #066875 !important;
}

/* ================= EXPANDERS ================= */

div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

div[data-testid="stExpander"] summary {
    background: #F1F5F9 !important;
    color: #102A43 !important;
}

div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] summary p {
    color: #102A43 !important;
    -webkit-text-fill-color: #102A43 !important;
    font-weight: 750 !important;
}

div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    padding: 20px !important;
}

/* ================= RESULT BANNERS ================= */

.result-banner {
    padding: 19px 22px;
    border-radius: 13px;
    margin-bottom: 20px;
    font-size: 17px;
    font-weight: 700;
}

.result-positive {
    background: #3B1820;
    color: #FCA5A5 !important;
    border: 1px solid #9F3445;
}

.result-negative {
    background: #103528;
    color: #86EFAC !important;
    border: 1px solid #23734C;
}

/* ================= RESULT CARDS ================= */

.result-card {
    background: #FFFFFF !important;
    padding: 23px;
    border-radius: 15px;
    min-height: 140px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 7px 20px rgba(0,0,0,0.15);
    overflow-wrap: anywhere;
}

.stApp .stMarkdown .result-card p,
.stApp .stHtml .result-card p {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    font-size: 15px !important;
    font-weight: 650 !important;
    margin-bottom: 15px !important;
}

.stApp .stMarkdown .result-card h2,
.stApp .stHtml .result-card h2 {
    color: #102A43 !important;
    -webkit-text-fill-color: #102A43 !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

.stApp .stMarkdown .result-card h3,
.stApp .stHtml .result-card h3 {
    color: #102A43 !important;
    -webkit-text-fill-color: #102A43 !important;
    font-size: 20px !important;
    font-weight: 750 !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}

/* ================= PROGRESS BAR ================= */

div[data-testid="stProgressBar"] {
    background: #334155 !important;
    border-radius: 20px !important;
    overflow: hidden;
}

div[data-testid="stProgressBar"] > div > div {
    background: #087E8B !important;
}

/* ================= CAPTIONS ================= */

.stApp [data-testid="stCaptionContainer"] p {
    color: #AEBCCB !important;
    font-size: 13px !important;
}

/* ================= DISCLAIMER ================= */

.disclaimer {
    background: #292719;
    border: 1px solid #75612B;
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
}

.disclaimer h3 {
    color: #FDE68A !important;
}

.disclaimer p {
    color: #FDE68A !important;
    line-height: 1.7;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    padding: 30px 15px;
    color: #94A3B8 !important;
    font-size: 13px;
}

/* ================= RESPONSIVE ================= */

@media (max-width: 768px) {

    .main .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 23px;
    }

    .hero h1 {
        font-size: 30px !important;
    }

    div[data-testid="stForm"] {
        padding: 16px !important;
    }

    .info-card {
        padding: 17px;
        min-height: 115px;
    }

    .info-card h2 {
        font-size: 30px !important;
    }

    .result-card {
        padding: 17px;
    }
}

</style>
""")


# ============================================================
# 5. LOAD MODELS AND DATASETS
# ============================================================

@st.cache_resource
def load_artifact(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


@st.cache_data
def load_dataset(data_path):

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}"
        )

    return pd.read_csv(data_path)


# ============================================================
# 6. MODEL ARTIFACT VALIDATION
# ============================================================

def get_model_information(artifact):

    if not isinstance(artifact, dict):
        raise ValueError(
            "The saved model artifact must be a dictionary."
        )

    required_keys = [
        "model",
        "threshold",
        "feature_columns"
    ]

    missing = [
        key for key in required_keys
        if key not in artifact
    ]

    if missing:
        raise ValueError(
            f"Model artifact is missing keys: {missing}"
        )

    return (
        artifact["model"],
        float(artifact["threshold"]),
        artifact["feature_columns"]
    )


# ============================================================
# 7. PATIENT INPUT FORM
# ============================================================

def create_input_form(dataset, feature_columns, disease):

    feature_values = {}

    render_html("""
    <div class="section-heading">
        Enter Patient Information
    </div>

    <div class="section-description">
        Enter the required numerical measurements.
        Default values are dataset medians for demonstration only,
        not actual patient measurements.
    </div>
    """)

    with st.form("patient_prediction_form"):

        if disease == "Breast Cancer":

            st.warning(
                "This dataset uses 30 numerical measurements "
                "derived from digitized breast-mass images. "
                "Default values are for demonstration only."
            )

            groups = {
                "Mean Measurements": [
                    c for c in feature_columns if c.endswith("1")
                ],

                "Standard Error Measurements": [
                    c for c in feature_columns if c.endswith("2")
                ],

                "Worst Measurements": [
                    c for c in feature_columns if c.endswith("3")
                ]
            }

            for group_name, columns in groups.items():

                with st.expander(
                    group_name,
                    expanded=True
                ):

                    col1, col2 = st.columns(2)

                    for index, feature in enumerate(columns):

                        default_value = float(
                            dataset[feature].median()
                        )

                        with (
                            col1 if index % 2 == 0 else col2
                        ):

                            feature_values[feature] = st.number_input(
                                feature.replace("_", " ").title(),
                                value=default_value,
                                format="%.4f",
                                key=f"{disease}_{feature}"
                            )

        else:

            col1, col2 = st.columns(2)

            for index, feature in enumerate(feature_columns):

                default_value = float(
                    dataset[feature].median()
                )

                valid_values = dataset[feature].dropna()

                is_integer = (
                    not valid_values.empty
                    and valid_values.mod(1).eq(0).all()
                )

                with (
                    col1 if index % 2 == 0 else col2
                ):

                    feature_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=default_value,
                        step=1.0 if is_integer else 0.1,
                        format="%.2f",
                        key=f"{disease}_{feature}"
                    )

        submitted = st.form_submit_button(
            "🔍 Predict",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        return pd.DataFrame(
            [feature_values],
            columns=feature_columns
        )

    return None


# ============================================================
# 8. DISPLAY PREDICTION
# ============================================================

def display_prediction(artifact, input_data, disease_config):

    model, threshold, _ = get_model_information(artifact)

    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain positive class 1."
        )

    positive_index = classes.index(1)

    positive_score = float(
        probabilities[positive_index]
    )

    prediction = int(
        positive_score >= threshold
    )

    render_html("""
    <hr>
    <div class="section-heading">
        Prediction Result
    </div>
    """)

    if prediction == 1:

        message = html.escape(
            disease_config["positive"]
        )

        render_html(f"""
        <div class="result-banner result-positive">
            ⚠️ {message}
        </div>
        """)

    else:

        message = html.escape(
            disease_config["negative"]
        )

        render_html(f"""
        <div class="result-banner result-negative">
            ✅ {message}
        </div>
        """)

    col1, col2, col3 = st.columns(3)

    with col1:

        render_html(f"""
        <div class="result-card">
            <p>Positive-class model score</p>
            <h2>{positive_score * 100:.2f}%</h2>
        </div>
        """)

    with col2:

        render_html(f"""
        <div class="result-card">
            <p>Decision Threshold</p>
            <h2>{threshold * 100:.2f}%</h2>
        </div>
        """)

    with col3:

        algorithm = html.escape(
            str(
                artifact.get(
                    "selected_model",
                    type(model).__name__
                )
            )
        )

        render_html(f"""
        <div class="result-card">
            <p>Selected Algorithm</p>
            <h3>{algorithm}</h3>
        </div>
        """)

    st.progress(
        positive_score,
        text="Positive-class model score"
    )

    st.caption(
        "The score is the model's estimated positive-class output. "
        "It is not necessarily a calibrated probability and is not "
        "a medical diagnosis."
    )


# ============================================================
# 9. SIDEBAR - DIRECT DISEASE SELECTION
# No dropdown. Only three selectable diseases.
# ============================================================

with st.sidebar:

    render_html("""
    <div style="padding-top: 12px;">
        <h1 style="color:#FFFFFF !important;">
            🩺 MediPredict AI
        </h1>

        <p style="color:#CBD5E1 !important;">
            Machine Learning Disease Prediction
        </p>
    </div>
    """)

    st.markdown("---")

    st.subheader("Available Diseases")

    selected_disease = st.radio(
        "Choose a disease",
        options=list(DISEASES.keys()),
        format_func=lambda disease: (
            f"{DISEASES[disease]['icon']}  {disease} Prediction"
        ),
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.info("Educational AIML Project")


# ============================================================
# 10. MAIN HERO SECTION
# ============================================================

render_html("""
<div class="hero">

    <h1>🩺 MediPredict AI</h1>

    <p>
        Intelligent disease prediction using Machine Learning.
        Explore three disease classification models through
        a unified interactive dashboard.
    </p>

</div>
""")


# ============================================================
# 11. DASHBOARD OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)

cards = [
    ("03", "Disease Datasets"),
    ("03", "ML Algorithms Compared"),
    ("09", "Original Model Experiments")
]

for column, (number, label) in zip(
    [col1, col2, col3],
    cards
):

    with column:

        render_html(f"""
        <div class="info-card">

            <h2>{number}</h2>

            <p>{label}</p>

        </div>
        """)


# ============================================================
# 12. SELECTED DISEASE
# ============================================================

st.markdown("---")

config = DISEASES[selected_disease]

render_html(f"""
<div class="section-heading">
    {config['icon']} {html.escape(selected_disease)} Prediction
</div>

<div class="section-description">
    {html.escape(config['description'])}
</div>
""")


# ============================================================
# 13. LOAD MODEL AND PREDICT
# ============================================================

try:

    artifact = load_artifact(
        config["model"]
    )

    dataset = load_dataset(
        config["data"]
    )

    model, threshold, feature_columns = get_model_information(
        artifact
    )

    missing_columns = [
        feature
        for feature in feature_columns
        if feature not in dataset.columns
    ]

    if missing_columns:

        st.error(
            f"Dataset is missing model features: {missing_columns}"
        )

        st.stop()

    st.success(
        f"{selected_disease} model loaded successfully."
    )

    input_data = create_input_form(
        dataset,
        feature_columns,
        selected_disease
    )

    if input_data is not None:

        display_prediction(
            artifact,
            input_data,
            config
        )


except FileNotFoundError as error:

    st.error(str(error))

except Exception as error:

    st.error(f"Application error: {error}")


# ============================================================
# 14. MEDICAL DISCLAIMER
# ============================================================

st.markdown("---")

render_html("""
<div class="disclaimer">

    <h3>⚕️ Medical Disclaimer</h3>

    <p>
        MediPredict AI is an educational academic machine
        learning prototype.
    </p>

    <p>
        Predictions must not be used to diagnose, screen,
        or treat any medical condition.
    </p>

    <p>
        The models may produce inaccurate or biased outputs
        and have not been clinically validated.
    </p>

    <p>
        Always consult a qualified healthcare professional
        for medical assessment and decisions.
    </p>

</div>
""")


# ============================================================
# 15. FOOTER
# ============================================================

render_html("""
<div class="footer">

    <p>
        MediPredict AI | Artificial Intelligence and Machine Learning
    </p>

    <p>
        Academic Project | Disease Classification
    </p>

</div>
""")
