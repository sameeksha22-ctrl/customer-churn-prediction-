from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Keep these two files in the same folder as app.py.
MODEL_PATH = Path("churn_prediction_pipeline.joblib")
METADATA_PATH = Path("churn_project_metadata.joblib")

st.set_page_config(
    page_title="ChurnSense | Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp {
    background:
      radial-gradient(circle at 8% 2%, rgba(99,102,241,.10), transparent 26%),
      radial-gradient(circle at 95% 10%, rgba(20,184,166,.10), transparent 25%),
      #f6f8fc;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
[data-testid="stSidebar"] { background: #10182f; }
[data-testid="stSidebar"] * { color: #f3f6ff !important; }
.hero {
    background: linear-gradient(120deg, #10182f 0%, #283b70 62%, #236b78 100%);
    padding: 2rem 2.2rem; border-radius: 22px; color: white;
    box-shadow: 0 14px 35px rgba(24, 39, 75, .16); margin-bottom: 1.2rem;
}
.hero h1 { color: white; margin: 0 0 .45rem 0; font-size: 2.35rem; }
.hero p { color: #dce7ff; margin: 0; font-size: 1.03rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 800; color: #7ce7d3; }
.metric-card {
    background: white; border: 1px solid #e6eaf2; border-radius: 16px;
    padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(24,39,75,.045);
}
.metric-label { color: #667085; font-size: .85rem; margin-bottom: .25rem; }
.metric-value { color: #17233f; font-size: 1.65rem; font-weight: 800; }
.section-title { color: #17233f; font-weight: 800; font-size: 1.25rem; margin: .3rem 0 .7rem 0; }
.note { background: #eef5ff; border-left: 4px solid #5478e8; padding: .85rem 1rem; border-radius: 8px; color: #344563; }
.result-box {
    padding: 1.25rem 1.4rem; border-radius: 16px; margin: .6rem 0 1rem 0;
    border: 1px solid #dce5ef; background: white;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4f46e5, #0f9f9a);
    border: 0; color: white; border-radius: 10px; font-weight: 750;
    padding: .7rem 1.2rem; min-height: 3rem;
}
.small-muted { color: #667085; font-size: .82rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata

def pretty_label(name):
    """Make raw dataset column names easier to read in the form."""
    return str(name).replace("_", " ").replace("-", " ").strip().title()

def get_category_options(model):
    """Read categories learned by the fitted OneHotEncoder from the saved pipeline."""
    options = {}
    try:
        preprocessor = model.named_steps["preprocessor"]
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name != "categorical" or transformer == "drop":
                continue
            encoder = transformer.named_steps["onehot"]
            for column, values in zip(columns, encoder.categories_):
                clean_values = []
                for value in values:
                    try:
                        if pd.isna(value):
                            continue
                    except (TypeError, ValueError):
                        pass
                    clean_values.append(value.item() if isinstance(value, np.generic) else value)
                # Remove duplicates while preserving order.
                options[column] = list(dict.fromkeys(clean_values))
    except Exception:
        # The app will still display a helpful message if a different pipeline is used.
        pass
    return options

def build_customer_form(metadata, category_options):
    feature_columns = metadata.get("feature_columns", [])
    numeric_features = set(metadata.get("numeric_features", []))
    customer = {}
    left, right = st.columns(2)
    for index, column in enumerate(feature_columns):
        target_col = left if index % 2 == 0 else right
        with target_col:
            label = pretty_label(column)
            if column in numeric_features:
                # Use a generic editable numeric field. Values should be realistic for the dataset.
                customer[column] = st.number_input(
                    label,
                    value=0.0,
                    step=1.0,
                    key=f"field_{index}",
                    help=f"Numeric input for the dataset feature: {column}",
                )
            else:
                choices = category_options.get(column, [])
                if choices:
                    display_choices = [str(x) for x in choices]
                    selected = st.selectbox(
                        label, options=choices, format_func=lambda x: str(x),
                        key=f"field_{index}",
                        help=f"Choose a value observed by the model during training: {column}",
                    )
                    customer[column] = selected
                else:
                    customer[column] = st.text_input(
                        label, value="", key=f"field_{index}",
                        help=f"Text input for the dataset feature: {column}",
                    )
    return pd.DataFrame([customer], columns=feature_columns)

def probability_for_churn(model, row):
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(row)[0]
    classifier = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    classes = list(getattr(classifier, "classes_", []))
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    if len(probabilities) == 2:
        return float(probabilities[1])
    return None

# ---------- Load and validate artifacts ----------
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Customer Retention Intelligence</div>
      <h1>ChurnSense</h1>
      <p>A machine-learning demo for understanding customer churn risk.</p>
    </div>
    """, unsafe_allow_html=True)
    st.error("The trained model files were not found.")
    st.markdown("""
    **Place these three files in the same folder:**
    1. `app.py` (this Streamlit interface)
    2. `churn_prediction_pipeline.joblib`
    3. `churn_project_metadata.joblib`

    Run every cell in your Jupyter notebook first. The final cell creates the two `.joblib` files.
    If you cannot find them, check Jupyter's current working folder.
    """)
    st.stop()

try:
    model, metadata = load_assets()
except Exception as exc:
    st.error(f"Could not load the saved model files: {exc}")
    st.info("Make sure the model files were generated by the corrected notebook and are in the same folder as app.py.")
    st.stop()

category_options = get_category_options(model)
metrics = metadata.get("test_metrics", {})
model_name = metadata.get("selected_model", "Trained classifier")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📊 ChurnSense")
    st.caption("Customer Retention Intelligence")
    st.markdown("---")
    st.markdown("### About this demo")
    st.write("Enter customer details and use the saved machine-learning pipeline to estimate churn risk.")
    st.markdown("### Model details")
    st.write(f"**Selected model:** {model_name}")
    st.write(f"**Input features:** {len(metadata.get('feature_columns', []))}")
    st.markdown("---")
    st.caption("Educational portfolio project • Predictions are estimates, not guarantees.")

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Machine Learning • Customer Analytics</div>
  <h1>ChurnSense</h1>
  <p>Turn customer data into a clearer view of churn risk — then use the insight to guide retention conversations.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_model, tab_about = st.tabs(["✨ Predict churn", "📈 Model snapshot", "ℹ️ How it works"])

with tab_predict:
    st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
    st.write("Fill in the available customer attributes, then run the prediction. Use values that make sense for the dataset you trained on.")
    with st.form("customer_form"):
        customer_df = build_customer_form(metadata, category_options)
        submitted = st.form_submit_button("✨ Estimate churn risk", type="primary", use_container_width=True)

    if submitted:
        try:
            prediction = int(model.predict(customer_df)[0])
            probability = probability_for_churn(model, customer_df)
            st.markdown("### Prediction result")
            if prediction == 1:
                st.markdown("""
                <div class="result-box">
                  <h3 style="color:#b42318;margin:0;">⚠️ Higher churn concern</h3>
                  <p style="color:#475467;margin:.45rem 0 0 0;">The model classifies this customer as likely to churn. Consider reviewing service experience and retention options.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box">
                  <h3 style="color:#067647;margin:0;">✅ Lower churn concern</h3>
                  <p style="color:#475467;margin:.45rem 0 0 0;">The model classifies this customer as unlikely to churn. Continue monitoring customer experience.</p>
                </div>
                """, unsafe_allow_html=True)
            if probability is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Estimated churn probability", f"{probability:.1%}")
                with col2:
                    st.progress(min(max(probability, 0.0), 1.0))
                    st.caption("This is a model-estimated probability, not a certainty.")
            st.markdown("#### Submitted customer details")
            st.dataframe(customer_df, use_container_width=True, hide_index=True)
            csv = customer_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download this input as CSV", data=csv, file_name="customer_prediction_input.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("Check that the notebook's saved model and metadata were generated successfully and match this app.")

with tab_model:
    st.markdown('<div class="section-title">Held-out test-set metrics</div>', unsafe_allow_html=True)
    st.write("These values are loaded from the evaluation performed by your notebook. They are not hard-coded.")
    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
    ]
    cols = st.columns(4)
    for col, (label, key) in zip(cols, metric_labels):
        value = metrics.get(key)
        with col:
            formatted = f"{float(value):.1%}" if value is not None else "N/A"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-value">{formatted}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.markdown('<div class="note">Interpret metrics together. For churn problems, recall and F1-score can be especially informative because missing a customer who will churn may be costly. The best metric depends on the business goal.</div>', unsafe_allow_html=True)
    st.markdown("#### Selected model")
    st.info(model_name)
    st.caption("These metrics reflect the saved notebook run and the particular dataset/split used. They do not guarantee the same performance on new customers.")

with tab_about:
    st.markdown('<div class="section-title">Project workflow</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Data preparation", "Identify the churn target, handle missing values, and exclude obvious ID fields."),
        ("02", "Preprocessing", "Impute missing values, scale numeric features, and one-hot encode categorical features."),
        ("03", "Model selection", "Compare Logistic Regression and Random Forest using cross-validation on the training split."),
        ("04", "Evaluation", "Evaluate the selected model on a held-out test set using accuracy, precision, recall, and F1-score."),
        ("05", "Prediction", "Apply the saved preprocessing-and-model pipeline to a customer profile."),
    ]
    for number, title, description in steps:
        st.markdown(
            f'<div class="metric-card" style="margin:.55rem 0;display:flex;gap:1rem;align-items:flex-start;">'
            f'<div style="font-weight:900;color:#4f46e5;font-size:1.2rem;">{number}</div>'
            f'<div><b style="color:#17233f;">{title}</b><div style="color:#667085;">{description}</div></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("")
    st.warning("Important limitation: the app reflects patterns in the training dataset. Predictions can be wrong and should not be used as the sole basis for decisions about real customers.")
