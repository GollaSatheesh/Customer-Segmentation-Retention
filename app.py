
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    retention = pd.read_csv(
        "project_data/retention_predictions.csv"
    )

    rfm = pd.read_csv(
        "project_data/historical_rfm.csv"
    )

    return retention, rfm


retention, rfm = load_data()


# ---------------- LOAD ML MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return model, scaler


model, scaler = load_model()


# ---------------- DASHBOARD HEADER ----------------

st.title("📊 Customer Intelligence Dashboard")

st.write(
    "Customer Segmentation, Inactivity Prediction "
    "and Retention Analysis"
)

st.divider()


# ---------------- KPI SECTION ----------------

col1, col2, col3, col4 = st.columns(4)

total_customers = len(rfm)
total_segments = rfm["Segment"].nunique()

predicted_inactive = (
    retention["Predicted_Status"].sum()
)

average_risk = (
    retention["Inactivity_Probability"].mean() * 100
)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("RFM Segments", total_segments)
col3.metric("Predicted Inactive", int(predicted_inactive))
col4.metric("Average Inactivity Risk", f"{average_risk:.1f}%")

st.divider()


# ---------------- RFM SEGMENTATION ----------------

st.header("1. Customer Segmentation")

segment_counts = rfm["Segment"].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Distribution")
    st.bar_chart(segment_counts)

with col2:
    st.subheader("Segment Summary")

    st.dataframe(
        segment_counts.rename("Customers"),
        use_container_width=True
    )

st.divider()


# ---------------- INACTIVITY PREDICTION ----------------

st.header("2. Customer Inactivity Prediction")

predicted_counts = (
    retention["Predicted_Status"]
    .map({0: "Likely Active", 1: "Predicted Inactive"})
    .value_counts()
)

st.subheader("Prediction Distribution")

st.bar_chart(predicted_counts)

st.write(
    "The model uses a 0.3 probability threshold "
    "to identify customers predicted to become inactive."
)

st.divider()


# ---------------- RISK BY RFM SEGMENT ----------------

st.header("3. Inactivity Risk by Customer Segment")

segment_risk = retention.groupby("Segment").agg(
    Total_Customers=("Customer ID", "count"),
    Predicted_Inactive=("Predicted_Status", "sum"),
    Average_Risk=("Inactivity_Probability", "mean")
)

segment_risk["Inactive_Percentage"] = (
    segment_risk["Predicted_Inactive"] /
    segment_risk["Total_Customers"] * 100
)

st.dataframe(
    segment_risk.round(2),
    use_container_width=True
)

st.bar_chart(
    segment_risk["Inactive_Percentage"]
)

st.divider()


# ---------------- AT-RISK CUSTOMER TABLE ----------------

st.header("4. At-Risk Customer Identification")

selected_segment = st.selectbox(
    "Select Customer Segment",
    ["All"] + sorted(
        retention["Segment"].dropna().unique().tolist()
    )
)

min_risk = st.slider(
    "Minimum Inactivity Probability",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

filtered_data = retention[
    retention["Inactivity_Probability"] >= min_risk
].copy()

if selected_segment != "All":
    filtered_data = filtered_data[
        filtered_data["Segment"] == selected_segment
    ]

filtered_data = filtered_data.sort_values(
    "Inactivity_Probability",
    ascending=False
)

st.dataframe(
    filtered_data[
        [
            "Customer ID",
            "Segment",
            "Frequency",
            "Monetary",
            "Recency",
            "Inactivity_Probability"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Predictions are based on historical customer behaviour "
    "and should be treated as risk estimates, not certainties."
)

st.divider()


# ---------------- NEW CUSTOMER PREDICTION ----------------

st.header("5. Predict Customer Inactivity")

st.write(
    "Enter customer purchasing details to predict "
    "their probability of becoming inactive."
)

with st.form("prediction_form"):

    frequency = st.number_input(
        "Purchase Frequency",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Total Spending",
        min_value=0.01,
        value=1000.0
    )

    recency = st.number_input(
        "Days Since Last Purchase",
        min_value=0,
        max_value=90,
        value=30
    )

    predict_button = st.form_submit_button(
        "Predict Inactivity"
    )

if predict_button:

    customer_data = pd.DataFrame(
        [[frequency, monetary, recency]],
        columns=["Frequency", "Monetary", "Recency"]
    )

    scaled_data = scaler.transform(customer_data)

    probability = model.predict_proba(scaled_data)[0][1]

    st.metric(
        "Inactivity Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.3:
        st.warning(
            "Customer is predicted to become inactive."
        )
    else:
        st.success(
            "Customer is predicted to remain active."
        )

    st.caption(
        "This is a model prediction, not a guarantee "
        "of future customer behaviour."
    )


st.divider()

st.success("Customer Intelligence Dashboard Loaded Successfully!")
