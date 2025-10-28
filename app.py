import json
from pathlib import Path

import joblib
import numpy as pd
import pandas as pd
import streamlit as st

# ---------------------------
# Basic Page Setup
# ---------------------------
st.set_page_config(page_title="Weather Conditions Predictor", page_icon="🌦️", layout="wide")
st.title("🌦️ Weather Conditions Predictor")
st.caption("Predict whether a day is **Dry**, **Rainy**, or **Snowy** using your local weather data.")

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "artifacts" / "weather_conditions_model.pkl"
REPORT_PATH = APP_DIR / "artifacts" / "report.json"
FIG_CM_PATH = APP_DIR / "figures" / "confusion_matrix.png"

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def try_read_csv(p: Path):
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.error(f"Couldn't read {p.name}: {e}")
    return None

def load_data():
    # Sidebar source picker
    st.sidebar.header("📥 Data Source")
    choice = st.sidebar.radio(
        "Choose data source",
        ["Auto-detect file", "Upload CSV"],
        help="Auto-detect searches for 'local_weather.csv' and 'local_weather (1).csv' next to app.py",
    )

    if choice == "Auto-detect file":
        candidates = [
            APP_DIR / "local_weather.csv",
            APP_DIR / "local_weather (1).csv",
        ]
        for p in candidates:
            df = try_read_csv(p)
            if df is not None:
                st.sidebar.success(f"Using file: {p.name}")
                return df, p.name
        st.sidebar.warning("No CSV found in the app folder. Switch to 'Upload CSV' below.")
        return None, None

    # Upload CSV
    up = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.sidebar.success("Uploaded CSV loaded.")
            return df, up.name
        except Exception as e:
            st.sidebar.error(f"Upload error: {e}")
            return None, None
    return None, None

def show_data_profile(df: pd.DataFrame, target_col="Condition"):
    st.subheader("🔎 Data Profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Target", target_col)

    with st.expander("View columns & types"):
        st.write(df.dtypes.to_frame("dtype"))

    with st.expander("Missing values (%)"):
        miss = (df.isna().mean() * 100).round(2)
        st.write(miss.sort_values(ascending=False).to_frame("% missing"))

    with st.expander("Quick stats (numeric)"):
        st.write(df.describe().T)

    if target_col in df.columns:
        st.subheader("Class Distribution")
        counts = df[target_col].astype(str).value_counts()
        st.bar_chart(counts)

def show_model_card():
    st.subheader("🧠 Model Card (from artifacts/report.json)")
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text())
            c1, c2, c3 = st.columns(3)
            c1.metric("Best baseline", str(report.get("best_baseline", "n/a")))
            c2.metric("Val Accuracy", f"{report.get('val_accuracy', 0):.3f}")
            c3.metric("Timestamp", report.get("timestamp", "n/a"))
            with st.expander("Raw report JSON"):
                st.json(report)
        except Exception as e:
            st.warning(f"Couldn't load report.json: {e}")
    else:
        st.info("No report.json found. Run your training notebook to generate it.")

def show_confusion_matrix():
    st.subheader("📷 Confusion Matrix")
    if FIG_CM_PATH.exists():
        st.image(str(FIG_CM_PATH), caption="Validation Confusion Matrix", use_column_width=True)
    else:
        st.info("No confusion_matrix.png found in /figures. Run your notebook to generate it.")

def build_row_from_defaults(df: pd.DataFrame, exclude=("Condition",)):
    """Return a single-row DataFrame with medians/modes for each column."""
    cols = [c for c in df.columns if c not in exclude]
    row = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            row[c] = float(df[c].median()) if df[c].notna().any() else 0.0
        else:
            row[c] = (df[c].mode(dropna=True).iloc[0] if not df[c].dropna().empty else "")
    return pd.DataFrame([row], columns=cols)

def try_feature_importance(pipeline):
    """Attempt to compute feature importance for tree-based final estimators."""
    try:
        model = pipeline.named_steps.get("model", None)
        prep = pipeline.named_steps.get("preprocess") or pipeline.named_steps.get("prep")
        if model is None or prep is None:
            return None

        # Try to derive feature names from preprocess
        cat_tf = prep.named_transformers_.get("cat")
        num_cols = prep.transformers_[0][2] if len(prep.transformers_) > 0 else []
        cat_cols = prep.transformers_[1][2] if len(prep.transformers_) > 1 else []

        # OneHot feature names
        if cat_tf is not None and hasattr(cat_tf.named_steps["oh"], "get_feature_names_out"):
            ohe_names = list(cat_tf.named_steps["oh"].get_feature_names_out(cat_cols))
        else:
            ohe_names = [f"{c}_OHE" for c in cat_cols]

        feature_names = list(num_cols) + ohe_names

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({"feature": feature_names[:len(importances)],
                                   "importance": importances[:len(feature_names)]})
            df_imp = df_imp.sort_values("importance", ascending=False).head(20)
            return df_imp
    except Exception:
        pass
    return None

# ---------------------------
# Load Model & Data
# ---------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model from {MODEL_PATH}. Error: {e}")
    st.stop()

df, data_name = load_data()
if df is None:
    st.stop()

# ---------------------------
# Main Layout Sections
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🧠 Model & Metrics", "🔮 Predict (Single)", "📦 Predict (Batch)"])

with tab1:
    st.write(f"**Data Source:** {data_name or 'Uploaded file'}")
    st.dataframe(df.head(10), use_container_width=True)
    show_data_profile(df, target_col="Condition")

with tab2:
    show_model_card()
    show_confusion_matrix()

    st.subheader("🌿 Feature Importance (Top 20)")
    fi = try_feature_importance(model)
    if fi is not None and not fi.empty:
        st.dataframe(fi, use_container_width=True)
        st.bar_chart(fi.set_index("feature")["importance"])
    else:
        st.info("Feature importance not available (this happens with non-tree models like Logistic Regression).")

with tab3:
    st.subheader("Fill Inputs Manually or Pick a Row to Auto-Fill")

    # Optional: user picks a row to auto-fill
    idx = st.number_input("Pick a row index from your data (optional)", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    use_row = st.checkbox("Use selected row to auto-fill inputs", value=False)

    # Build a baseline single row
    single = build_row_from_defaults(df)

    # Overwrite with a selected row if requested
    if use_row:
        row = df.drop(columns=["Condition"], errors="ignore").iloc[int(idx)].to_dict()
        single.iloc[0] = row

    # Editable inputs for common columns
    common_cols = ["TAVG", "TMAX", "TMIN", "PRCP", "SNOW", "WSF2"]
    cols_present = [c for c in common_cols if c in single.columns]
    if cols_present:
        st.caption("Tweak these common features (only those present in your data will appear):")
        col_widgets = st.columns(len(cols_present))
        for i, c in enumerate(cols_present):
            val = float(single.iloc[0][c]) if pd.notna(single.iloc[0][c]) else 0.0
            single.at[0, c] = col_widgets[i].number_input(c, value=val)

    if st.button("Predict Weather Condition"):
        try:
            pred = model.predict(single)[0]
            st.success(f"🌤️ Predicted Weather Condition: **{pred}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

with tab4:
    st.subheader("Batch Predictions (Upload CSV)")
    st.caption("Upload a CSV with the same columns your model expects (like your training data, without the 'Condition' target).")
    up = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
    if up is not None:
        try:
            new_df = pd.read_csv(up)
            st.write("Preview:", new_df.head())
            if st.button("Run Batch Prediction"):
                preds = model.predict(new_df)
                out = new_df.copy()
                out["Predicted_Condition"] = preds
                st.write("Results (first 20):")
                st.dataframe(out.head(20), use_container_width=True)

                # Download button
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="weather_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

st.markdown("---")
st.caption("Created by Gloria Ighagbon • Week 10 • Streamlit Demo")
