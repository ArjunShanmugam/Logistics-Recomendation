# app.py
import streamlit as st
import pandas as pd
from logistics_ml import load_and_preprocess_data, recommend_best_supplier
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Supplier Recommendation System",
    page_icon="🚚",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS (modern / glassmorphism style)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    [data-testid="stAppViewContainer"] > .main {
        background: radial-gradient(circle at top left, #1f2933, #020617);
        color: #e5e7eb;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #111827);
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Hero title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    /* Card container */
    .glass-card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 22px 50px rgba(15, 23, 42, 0.9);
        border-color: rgba(129, 140, 248, 0.55);
    }

    /* Supplier rank header */
    .rank-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.9rem;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.15);
        color: #bfdbfe;
        margin-bottom: 0.75rem;
    }

    /* Metric label + value */
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e5e7eb;
    }

    /* Progress bar override */
    .stProgress > div > div {
        background-image: linear-gradient(90deg, #22c55e, #3b82f6);
    }

    /* Small fade-in animation */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeInUp 0.35s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Cached loaders
# --------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("logistics_model.joblib")
    encoder = joblib.load("logistics_encoder.joblib")
    return model, encoder


@st.cache_data
def load_data_and_customers():
    df = load_and_preprocess_data("Transportation and Logistics Tracking Dataset.csv")
    # Make sure we have customerNameCode in df
    customers = (
        df["customerNameCode"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return df, customers


# --------------------------------------------------
# Load stuff
# --------------------------------------------------
model, encoder = load_models()
df, customer_options = load_data_and_customers()

# --------------------------------------------------
# Sidebar – compact & clean
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model Info")
    st.write(
        "This app recommends the best transport **supplier** for a given **customer**, "
        "based on historical on-time performance and your trained ML model."
    )
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Pick a customer from the dropdown\n"
                "2. Click **Recommend suppliers**\n"
                "3. Explore scores & metrics")
    st.markdown("---")
    st.caption("Built with Streamlit · ML + Logistics 🚚")

# --------------------------------------------------
# Main layout
# --------------------------------------------------
hero_col, _ = st.columns([3, 1])
with hero_col:
    st.markdown('<div class="hero-title">Supplier Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Smart matching of customers and transporters using historical performance & machine learning.</div>',
        unsafe_allow_html=True,
    )

# Tabs for future expansion if needed
tab_reco, tab_overview = st.tabs(["🔍 Recommendation", "📊 Overview"])

# --------------------------------------------------
# Recommendation tab
# --------------------------------------------------
with tab_reco:
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)

        st.markdown("#### 🎯 Select Customer")
        st.write("Start by choosing a customer. The dropdown is searchable, so you don’t have to remember exact names.")

        # Preselect something common if exists
        default_index = 0
        default_name = "Lucas tvs ltd"
        if default_name in customer_options:
            default_index = customer_options.index(default_name)

        with st.form("reco_form", clear_on_submit=False):
            selected_customer = st.selectbox(
                "Customer",
                customer_options,
                index=default_index,
                key="customer_select",
                help="Type to search – e.g., 'Ashok', 'Lucas', etc.",
            )

            submitted = st.form_submit_button("🚚 Recommend suppliers")

        if submitted:
            if not selected_customer:
                st.warning("Please select a customer to get recommendations.")
            else:
                with st.spinner("Running model and ranking suppliers…"):
                    try:
                        recommendations = recommend_best_supplier(
                            selected_customer, df, model, encoder
                        )

                        if isinstance(recommendations, str):
                            st.warning(recommendations)
                        elif not recommendations:
                            st.warning("No suppliers found for this customer.")
                        else:
                            st.success(
                                f"Top {len(recommendations)} suppliers for **{selected_customer}**"
                            )

                            # Show each supplier in its own glass card
                            for i, rec in enumerate(recommendations, start=1):
                                st.markdown("</div>", unsafe_allow_html=True)  # close previous card
                                st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)

                                st.markdown(
                                    f'<div class="rank-pill">#{i} · {rec["supplier"]}</div>',
                                    unsafe_allow_html=True,
                                )

                                c1, c2, c3 = st.columns(3)

                                with c1:
                                    st.markdown('<div class="metric-label">Predicted on-time</div>', unsafe_allow_html=True)
                                    st.markdown(
                                        f'<div class="metric-value">{rec["predicted_ontime"]*100:.1f}%</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.progress(float(rec["predicted_ontime"]))

                                with c2:
                                    st.markdown('<div class="metric-label">Historical on-time</div>', unsafe_allow_html=True)
                                    hist = rec.get("historical_ontime", 0)
                                    st.markdown(
                                        f'<div class="metric-value">{hist*100:.1f}%</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.progress(float(hist))

                                with c3:
                                    st.markdown('<div class="metric-label">Total trips</div>', unsafe_allow_html=True)
                                    st.markdown(
                                        f'<div class="metric-value">{int(rec["trip_count"])}</div>',
                                        unsafe_allow_html=True,
                                    )

                                st.caption(
                                    "Recommended based on highest predicted on-time probability, "
                                    "supported by historical performance and trip volume."
                                )

                            st.markdown("</div>", unsafe_allow_html=True)  # close last card

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        st.markdown("</div>", unsafe_allow_html=True)  # close main card

    # Right column – overview / quick stats
    with right:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("#### ⭐ Quick Snapshot")

        try:
            # If your preprocessing created an 'is_ontime' column, use it
            if "is_ontime" in df.columns:
                total_trips = len(df)
                overall_ontime = df["is_ontime"].mean()

                st.markdown('<div class="metric-label">Overall on-time rate</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{overall_ontime*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )
                st.progress(float(overall_ontime))

                st.markdown('<div class="metric-label" style="margin-top:0.8rem;">Total trips in dataset</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{total_trips}</div>',
                    unsafe_allow_html=True,
                )

            # Simple top suppliers leaderboard
            if "supplierNameCode" in df.columns and "customerNameCode" in df.columns:
                st.markdown("---")
                st.markdown("**Top suppliers by trip volume**")
                leaderboard = (
                    df.groupby("supplierNameCode")
                    .size()
                    .reset_index(name="trips")
                    .sort_values("trips", ascending=False)
                    .head(5)
                )
                st.dataframe(
                    leaderboard,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception:
            st.caption("Stats not available – check preprocessing.")
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Overview tab (optional extra content)
# --------------------------------------------------
with tab_overview:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 📊 Project Overview")
    st.write(
        """
        This dashboard demonstrates a **machine learning–powered supplier recommendation system** 
        built on a transportation & logistics dataset.
        
        **Key ideas:**
        - Each past trip is labelled as *on-time* or *delayed*.
        - A model learns patterns across customer, supplier, route, distance, and vehicle type.
        - For any selected customer, suppliers are ranked by:
            - Predicted on-time probability (ML model output)
            - Historical on-time rate
            - Number of trips (reliability of evidence)
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
