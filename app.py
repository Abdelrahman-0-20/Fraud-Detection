import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import io

# 
# Page config
# 
st.set_page_config(
    page_title="Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 
# Dark theme CSS
# 
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding: 2rem 2rem 3rem; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="metric-container"] {
        background: #1c2333;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 0.5rem;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #58a6ff;
        border-left: 4px solid #58a6ff;
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    .info-box {
        background: #1c2333;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .info-box b { color: #58a6ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #1c2333;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #8b949e;
        padding: 0.3rem 1.2rem;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #58a6ff !important;
        color: #0e1117 !important;
        font-weight: 600;
    }
    .stButton > button {
        background: #58a6ff;
        color: #0e1117;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.4rem 1.2rem;
    }
    .stButton > button:hover { background: #79c0ff; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; background: #161b22; }
    label { color: #8b949e !important; font-size: 0.85rem; }
    hr { border-color: #30363d; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# 
# Data loading – tries local CSV, falls back to synthetic
# 
@st.cache_data
def load_transaction_data():
    """Load transaction data from default CSV or generate synthetic."""
    try:
        df = pd.read_csv("AIML Dataset.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("fraud_data.csv")
        except FileNotFoundError:
            st.warning("No CSV found – using synthetic sample data for demonstration.")
            np.random.seed(42)
            n = 5000
            types = np.random.choice(
                ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'CASH_IN'],
                n, p=[0.2, 0.2, 0.3, 0.1, 0.2]
            )
            amounts = np.random.exponential(scale=5000, size=n) + 100
            old_orig = np.random.uniform(1000, 100000, n)
            new_orig = old_orig - amounts
            new_orig = np.maximum(new_orig, 0)
            old_dest = np.random.uniform(0, 50000, n)
            new_dest = old_dest + amounts
            is_fraud = np.zeros(n, dtype=int)
            fraud_candidates = np.where((types == 'CASH_OUT') | (types == 'TRANSFER'))[0]
            fraud_idx = np.random.choice(fraud_candidates, size=int(n*0.01), replace=False)
            is_fraud[fraud_idx] = 1
            for idx in fraud_idx:
                old_orig[idx] = np.random.uniform(1000, 50000)
                new_orig[idx] = 0
                old_dest[idx] = np.random.uniform(0, 1000)
                new_dest[idx] = old_dest[idx] + amounts[idx]
            df = pd.DataFrame({
                'type': types,
                'amount': amounts,
                'oldbalanceOrg': old_orig,
                'newbalanceOrig': new_orig,
                'oldbalanceDest': old_dest,
                'newbalanceDest': new_dest,
                'isFraud': is_fraud,
                'isFlaggedFraud': 0
            })

    # Feature engineering
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["balanceDiffOrig"]  = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"]  = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amountToOldBalOrig"] = np.where(
        df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0
    )
    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])
    return df

# 
# Helper to get a sample for EDA (cached)
# 
@st.cache_data
def get_eda_sample(df, sample_size=20000):
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df

# 
# Optimised model training – faster and parallel
# 
@st.cache_resource
def train_fraud_model(data, model_choice, sample_frac=1.0):
    """Train a model on a (possibly sampled) fraction of the data."""
    if sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42)

    feature_cols = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "type_enc",
        "errorBalanceOrig", "errorBalanceDest",
        "balanceDiffOrig", "balanceDiffDest", "amountToOldBalOrig"
    ]
    X = data[feature_cols]
    y = data["isFraud"]

    stratify = y if y.sum() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Faster models with parallel processing
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
    }
    clf = classifiers[model_choice]

    with st.spinner(f"Training {model_choice} ... (this may take a moment)"):
        clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    return clf, scaler, X_test, y_test, y_pred, y_proba, feature_cols

# 
# Load data
# 
df = load_transaction_data()

# 
# Sidebar navigation
# 
with st.sidebar:
    st.markdown("##  Fraud Detection")
    st.markdown("---")
    page = st.selectbox(
        "Navigation",
        ["Case Study", "Raw & Export", "EDA & Visualizations", "ML Model"],
        index=0
    )
    st.markdown("---")
    st.caption("Financial Fraud Detection  |  ML App")
    st.caption("(EDA uses a random sample for speed)")

# 
# PAGE 1 – CASE STUDY
# 
if page == "Case Study":
    st.title(" Case Study — Financial Fraud Detection")
    st.markdown("---")

    total_tx = len(df)
    fraud_count = df["isFraud"].sum()
    fraud_rate = fraud_count / total_tx * 100 if total_tx > 0 else 0
    total_volume = df["amount"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("Fraudulent", f"{int(fraud_count):,}", delta_color="inverse")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Total Volume", f"${total_volume:,.0f}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Problem Statement</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Financial fraud costs billions every year. This dataset contains mobile money
        transactions labelled as fraudulent or genuine. The aim is to build a model
        that detects fraud early, with as few false alarms as possible.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Key Observations</div>', unsafe_allow_html=True)
        insights = [
            ("Fraud types", "Only TRANSFER and CASH_OUT transactions are fraudulent."),
            ("Sender balance", "In fraud cases, the sender’s balance drops to zero."),
            ("Balance error", "A mismatch between expected and actual balance is a strong signal."),
            ("Destination", "Fraudulent transfers show no increase in the destination balance."),
            ("Flagging", "The built‑in flag misses most frauds – we need a better model."),
        ]
        for title, desc in insights:
            st.markdown(f'<div class="info-box"><b>{title}:</b> {desc}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Proposed Workflow</div>', unsafe_allow_html=True)
        workflow = [
            ("1. Ingest", "Load raw transaction logs and check schema."),
            ("2. Engineer", "Create balance error, difference, and ratio features."),
            ("3. Train", "Fit Random Forest / Gradient Boosting with class balancing."),
            ("4. Evaluate", "Measure precision, recall, F1, and ROC‑AUC."),
            ("5. Deploy", "Serve predictions through this Streamlit dashboard."),
            ("6. Monitor", "Watch for model drift and retrain as needed."),
        ]
        for step, desc in workflow:
            st.markdown(f'<div class="info-box"><b>{step}:</b> {desc}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Target Metrics</div>', unsafe_allow_html=True)
        target_df = pd.DataFrame({
            "Metric": ["Recall (Fraud)", "Precision (Fraud)", "F1‑Score", "ROC‑AUC", "False Positive Rate"],
            "Goal": ["> 90%", "> 85%", "> 87%", "> 95%", "< 5%"]
        })
        st.dataframe(target_df, use_container_width=True, hide_index=True)

# 
# PAGE 2 – RAW DATA & EXPORT
# 
elif page == "Raw & Export":
    st.title(" Raw Data & Export")
    st.markdown("---")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        tx_types = ["All"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Transaction Type", tx_types)
    with filter_col2:
        fraud_filter = st.selectbox("Fraud Status", ["All", "Fraud Only", "Legitimate Only"])
    with filter_col3:
        min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
        amt_range = st.slider("Amount Range", min_amt, max_amt, (min_amt, max_amt))

    filtered_df = df.copy()
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["type"] == selected_type]
    if fraud_filter == "Fraud Only":
        filtered_df = filtered_df[filtered_df["isFraud"] == 1]
    elif fraud_filter == "Legitimate Only":
        filtered_df = filtered_df[filtered_df["isFraud"] == 0]
    filtered_df = filtered_df[
        (filtered_df["amount"] >= amt_range[0]) &
        (filtered_df["amount"] <= amt_range[1])
    ]

    st.markdown(f"**{len(filtered_df):,} rows** after filtering")
    st.dataframe(filtered_df.head(500), use_container_width=True)

    with st.expander("Export Data", expanded=False):
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            csv_data = filtered_df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv_data, "fraud_filtered.csv", "text/csv")
        with exp_col2:
            json_data = filtered_df.to_json(orient="records").encode()
            st.download_button("Download JSON", json_data, "fraud_filtered.json", "application/json")
        with exp_col3:
            buffer = io.BytesIO()
            filtered_df.to_excel(buffer, index=False, engine="openpyxl")
            st.download_button("Download Excel", buffer.getvalue(), "fraud_filtered.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("Data Summary", expanded=False):
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.write("**Shape:**", filtered_df.shape)
            st.write("**Column Types:**")
            st.dataframe(filtered_df.dtypes.rename("dtype").reset_index(), use_container_width=True, hide_index=True)
        with sum_col2:
            st.write("**Descriptive Statistics:**")
            st.dataframe(filtered_df.describe().T, use_container_width=True)

# 
# PAGE 3 – EDA & VISUALISATIONS (using sampled data for speed)
# 
elif page == "EDA & Visualizations":
    st.title(" EDA & Visualizations")
    st.markdown("---")

    if df.empty:
        st.warning("No data to display.")
    else:
        sample_df = get_eda_sample(df, sample_size=5000)
        st.caption(f"Plots based on a random sample of {len(sample_df):,} rows (for performance).")

        tab_dist, tab_fraud, tab_corr, tab_balance = st.tabs([
            "Distribution", "Fraud Analysis", "Correlation", "Balance Analysis"
        ])

        colour_palette = px.colors.qualitative.Set2

        # ---- Distribution ----
        with tab_dist:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(sample_df, names="type", title="Transaction Type Distribution",
                             color_discrete_sequence=colour_palette, hole=0.4)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(sample_df, x="amount", color="type", nbins=40,
                                   title="Amount Distribution by Type",
                                   color_discrete_sequence=colour_palette, barmode="overlay")
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14)
                st.plotly_chart(fig, use_container_width=True)

            fig = px.box(sample_df, x="type", y="amount", color="type",
                         title="Amount Statistics by Transaction Type",
                         color_discrete_sequence=colour_palette)
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font_color="#e6edf3", title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        # ---- Fraud Analysis ----
        with tab_fraud:
            fraud_by_type = sample_df.groupby("type")["isFraud"].sum().reset_index()
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(fraud_by_type, x="type", y="isFraud",
                             title="Fraud Count by Transaction Type",
                             color="type", color_discrete_sequence=colour_palette)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fraud_rate_by_type = sample_df.groupby("type")["isFraud"].mean().reset_index()
                fraud_rate_by_type["isFraud"] = fraud_rate_by_type["isFraud"] * 100
                fig = px.bar(fraud_rate_by_type, x="type", y="isFraud",
                             title="Fraud Rate (%) by Transaction Type",
                             color="type", color_discrete_sequence=colour_palette)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14,
                                  showlegend=False, yaxis_title="Fraud Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

            fraud_only = sample_df[sample_df["isFraud"] == 1]
            if not fraud_only.empty:
                fig = px.violin(fraud_only, x="type", y="amount", color="type",
                                title="Fraud Amount Distribution by Type",
                                color_discrete_sequence=colour_palette, box=True)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fraudulent transactions to show.")

        # ---- Correlation ----
        with tab_corr:
            numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
                            "oldbalanceDest", "newbalanceDest",
                            "errorBalanceOrig", "errorBalanceDest", "isFraud"]
            corr_matrix = sample_df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", title="Correlation Heatmap",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              font_color="#e6edf3", title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(sample_df, x="amount", y="balanceDiffOrig",
                                 color=sample_df["isFraud"].astype(str),
                                 title="Amount vs Balance Diff (Origin)",
                                 color_discrete_map={"0": "#58a6ff", "1": "#f85149"},
                                 labels={"color": "isFraud"}, opacity=0.6)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(sample_df, x="amount", y="balanceDiffDest",
                                 color=sample_df["isFraud"].astype(str),
                                 title="Amount vs Balance Diff (Destination)",
                                 color_discrete_map={"0": "#58a6ff", "1": "#f85149"},
                                 labels={"color": "isFraud"}, opacity=0.6)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14)
                st.plotly_chart(fig, use_container_width=True)

        # ---- Balance Analysis ----
        with tab_balance:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(sample_df, x="errorBalanceOrig", color=sample_df["isFraud"].astype(str),
                                   nbins=40, title="Balance Error (Origin) by Fraud Status",
                                   color_discrete_map={"0": "#58a6ff", "1": "#f85149"},
                                   labels={"color": "isFraud"}, barmode="overlay", opacity=0.7)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(sample_df, x="errorBalanceDest", color=sample_df["isFraud"].astype(str),
                                   nbins=40, title="Balance Error (Destination) by Fraud Status",
                                   color_discrete_map={"0": "#58a6ff", "1": "#f85149"},
                                   labels={"color": "isFraud"}, barmode="overlay", opacity=0.7)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3", title_font_size=14)
                st.plotly_chart(fig, use_container_width=True)

            fig = px.box(sample_df, x=sample_df["isFraud"].astype(str), y="amountToOldBalOrig",
                         color=sample_df["isFraud"].astype(str),
                         title="Amount / Old Balance Ratio by Fraud Status",
                         color_discrete_map={"0": "#58a6ff", "1": "#f85149"},
                         labels={"x": "isFraud"})
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font_color="#e6edf3", title_font_size=14, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# 
# PAGE 4 – MACHINE LEARNING MODEL
# 
elif page == "ML Model":
    st.title(" Machine Learning Model")
    st.markdown("---")

    tab_train, tab_metrics, tab_predict = st.tabs(["Train & Evaluate", "Model Metrics", "Predict"])

    # ---- TAB 1: Train ----
    with tab_train:
        left_col, right_col = st.columns([1, 2])
        with left_col:
            model_choice = st.selectbox(
                "Choose Model",
                ["Random Forest", "Gradient Boosting", "Logistic Regression"],
                key="model_choice"
            )
            sample_frac = st.slider(
                "Training data fraction",
                min_value=0.1, max_value=1.0, value=1.0, step=0.1,
                help="Use a smaller fraction for faster training (e.g., 0.5 = 50% of data)."
            )
            train_button = st.button(" Train Model")

        if train_button:
            with st.spinner(f"Training {model_choice} ..."):
                result = train_fraud_model(df, model_choice, sample_frac)
            st.session_state["model_result"] = result
            st.session_state["model_trained"] = True
            st.session_state["model_name"] = model_choice
            st.success(" Training complete!")

        if st.session_state.get("model_trained", False):
            clf, scaler, X_test, y_test, y_pred, y_proba, features = st.session_state["model_result"]
            report = classification_report(y_test, y_pred, output_dict=True)

            with right_col:
                st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{report.get('accuracy', 0):.3f}")
                fraud_metrics = report.get('1', report.get('1.0', {}))
                m2.metric("Precision (Fraud)", f"{fraud_metrics.get('precision', 0):.3f}")
                m3.metric("Recall (Fraud)", f"{fraud_metrics.get('recall', 0):.3f}")
                m4.metric("F1‑Score (Fraud)", f"{fraud_metrics.get('f1-score', 0):.3f}")

            with st.expander("Full Classification Report", expanded=False):
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df, use_container_width=True)
        else:
            with right_col:
                st.info(" Pick a model, set fraction, and click **Train Model** to start.")

    # ---- TAB 2: Metrics ----
    with tab_metrics:
        if not st.session_state.get("model_trained", False):
            st.info("Please train a model in the **Train & Evaluate** tab first.")
        else:
            clf, scaler, X_test, y_test, y_pred, y_proba, features = st.session_state["model_result"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True,
                                x=["Predicted Legit", "Predicted Fraud"],
                                y=["Actual Legit", "Actual Fraud"],
                                color_continuous_scale="Blues")
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"AUC = {roc_auc:.3f}",
                                         line=dict(color="#58a6ff", width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color="#8b949e", dash="dash"),
                                         showlegend=False))
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3",
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate")
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
                if hasattr(clf, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "Feature": features,
                        "Importance": clf.feature_importances_
                    }).sort_values("Importance", ascending=True)
                    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Blues")
                    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                      font_color="#e6edf3", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")

            with col4:
                st.markdown('<div class="section-header">Precision‑Recall Curve</div>', unsafe_allow_html=True)
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                         line=dict(color="#3fb950", width=2)))
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#e6edf3",
                                  xaxis_title="Recall",
                                  yaxis_title="Precision")
                st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3: Predict ----
    with tab_predict:
        if not st.session_state.get("model_trained", False):
            st.info("Please train a model in the **Train & Evaluate** tab first.")
        else:
            clf, scaler, X_test, y_test, y_pred, y_proba, features = st.session_state["model_result"]

            st.markdown('<div class="section-header">Enter Transaction Details</div>', unsafe_allow_html=True)

            input_col1, input_col2, input_col3 = st.columns(3)

            with input_col1:
                tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
                amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
                old_balance_orig = st.number_input("Old Balance (Origin)", min_value=0.0, value=50000.0, step=1000.0)

            with input_col2:
                new_balance_orig = st.number_input("New Balance (Origin)", min_value=0.0, value=40000.0, step=1000.0)
                old_balance_dest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0, step=1000.0)
                new_balance_dest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0, step=1000.0)

            is_valid = True
            if amount < 0:
                st.error("Amount cannot be negative.")
                is_valid = False
            if any(v < 0 for v in [new_balance_orig, new_balance_dest, old_balance_orig, old_balance_dest]):
                st.error("Balances cannot be negative.")
                is_valid = False

            type_encoder = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
            err_orig = new_balance_orig + amount - old_balance_orig
            err_dest = old_balance_dest + amount - new_balance_dest
            diff_orig = old_balance_orig - new_balance_orig
            diff_dest = new_balance_dest - old_balance_dest
            ratio = amount / old_balance_orig if old_balance_orig > 0 else 0

            input_vector = np.array([[
                amount, old_balance_orig, new_balance_orig,
                old_balance_dest, new_balance_dest,
                type_encoder.get(tx_type, 0),
                err_orig, err_dest, diff_orig, diff_dest, ratio
            ]])

            with input_col3:
                st.markdown("")
                st.markdown("")
                predict_button = st.button("Run Prediction", disabled=not is_valid)

            if predict_button and is_valid:
                input_scaled = scaler.transform(input_vector)
                prediction = clf.predict(input_scaled)[0]
                probability = clf.predict_proba(input_scaled)[0][1]

                st.markdown("---")
                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    if prediction == 1:
                        st.error(f" FRAUD DETECTED  —  Probability: {probability:.1%}")
                    else:
                        st.success(f" LEGITIMATE  —  Fraud Probability: {probability:.1%}")

                    details_df = pd.DataFrame({
                        "Feature": features,
                        "Value": input_vector[0]
                    })
                    st.dataframe(details_df, use_container_width=True, hide_index=True)

                with result_col2:
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(probability * 100, 1),
                        title={"text": "Fraud Risk Score", "font": {"color": "#e6edf3"}},
                        number={"suffix": "%", "font": {"color": "#e6edf3"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                            "bar": {"color": "#f85149" if prediction == 1 else "#3fb950"},
                            "bgcolor": "#161b22",
                            "steps": [
                                {"range": [0, 30], "color": "#161b22"},
                                {"range": [30, 70], "color": "#30363d"},
                                {"range": [70, 100], "color": "#3d1f1f"},
                            ],
                            "threshold": {
                                "line": {"color": "#f85149", "width": 3},
                                "thickness": 0.75,
                                "value": 70
                            }
                        }
                    ))
                    gauge_fig.update_layout(paper_bgcolor="#0e1117", font_color="#e6edf3", height=300)
                    st.plotly_chart(gauge_fig, use_container_width=True)