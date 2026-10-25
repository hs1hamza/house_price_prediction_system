import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Prediction")
st.write(
    "Enter the house features used by the Assignment 3 model to estimate the sale price."
)
st.info(
    "This model was trained on preprocessed features. Some fields below are engineered "
    "or transformed values (for example, log features), not raw measurements. Use values "
    "prepared in the same way as the training dataset for meaningful predictions."
)

MODEL_PATH = Path(__file__).with_name("house_price_prediction.pkl")

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Put house_price_prediction.pkl in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

# These are the exact 20 feature columns used by the Assignment 3 model.
# Defaults are taken from a representative row in the assignment notebook.
with st.form("house_features"):
    st.subheader("House features")

    col1, col2 = st.columns(2)

    with col1:
        OverallQual = st.number_input("OverallQual (overall quality, 1–10)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
        ExterQual_TA = st.selectbox("ExterQual_TA", [0, 1], index=0, help="One-hot encoded exterior quality indicator.")
        GrLivArea = st.number_input("GrLivArea (transformed)", value=8.35, step=0.01, format="%.6f")
        GrLivArea_log_sq = st.number_input("GrLivArea_log_sq", value=5.00, step=0.01, format="%.6f")
        GrLivArea_log = st.number_input("GrLivArea_log", value=2.23, step=0.01, format="%.6f")
        GarageCars_log_sq = st.number_input("GarageCars_log_sq", value=1.21, step=0.01, format="%.6f")
        Total_Home_Quality = st.number_input("Total_Home_Quality", value=10.99, step=0.1, format="%.6f")
        KitchenQual_Ex = st.selectbox("KitchenQual_Ex", [0, 1], index=0)
        GarageCars_log = st.number_input("GarageCars_log", value=1.10, step=0.01, format="%.6f")
        Total_Bathrooms = st.number_input("Total_Bathrooms (transformed)", min_value=0.0, value=2.35, step=0.1, format="%.6f")

    with col2:
        GarageCars = st.number_input("GarageCars", min_value=0.0, max_value=4.0, value=2.0, step=1.0)
        GarageType_Attchd = st.selectbox("GarageType_Attchd", [0, 1], index=1)
        LotArea = st.number_input("LotArea (transformed)", value=13.83, step=0.1, format="%.6f")
        Neighborhood_Crawfor = st.selectbox("Neighborhood_Crawfor", [0, 1], index=0)
        KitchenQual_Gd = st.selectbox("KitchenQual_Gd", [0, 1], index=1)
        GarageType_Detchd = st.selectbox("GarageType_Detchd", [0, 1], index=0)
        LandSlope_Gtl = st.selectbox("LandSlope_Gtl", [0, 1], index=1)
        YrBltAndRemod = st.number_input("YrBltAndRemod", value=4000.0, step=1.0)
        TotRmsAbvGrd = st.number_input("TotRmsAbvGrd (transformed)", value=2.26, step=0.01, format="%.6f")
        MSZoning_RM = st.selectbox("MSZoning_RM", [0, 1], index=0)

    submitted = st.form_submit_button("Predict house price", type="primary", use_container_width=True)

if submitted:
    # Build the input DataFrame in the exact order expected by the saved model.
    input_data = pd.DataFrame([{
        "OverallQual": OverallQual,
        "ExterQual_TA": ExterQual_TA,
        "GrLivArea": GrLivArea,
        "GrLivArea_log_sq": GrLivArea_log_sq,
        "GrLivArea_log": GrLivArea_log,
        "GarageCars_log_sq": GarageCars_log_sq,
        "Total_Home_Quality": Total_Home_Quality,
        "KitchenQual_Ex": KitchenQual_Ex,
        "GarageCars_log": GarageCars_log,
        "Total_Bathrooms": Total_Bathrooms,
        "GarageCars": GarageCars,
        "GarageType_Attchd": GarageType_Attchd,
        "LotArea": LotArea,
        "Neighborhood_Crawfor": Neighborhood_Crawfor,
        "KitchenQual_Gd": KitchenQual_Gd,
        "GarageType_Detchd": GarageType_Detchd,
        "LandSlope_Gtl": LandSlope_Gtl,
        "YrBltAndRemod": YrBltAndRemod,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "MSZoning_RM": MSZoning_RM,
    }])

    try:
        # Reorder columns using the model's saved feature names when available.
        expected_columns = getattr(model, "feature_names_in_", None)
        if expected_columns is not None:
            input_data = input_data[list(expected_columns)]

        predicted_price = float(model.predict(input_data)[0])
        st.success(f"Estimated sale price: ${predicted_price:,.2f}")
        st.caption("This is a machine-learning estimate, not a formal property valuation.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption("Check that the model file matches the Assignment 3 feature columns.")
