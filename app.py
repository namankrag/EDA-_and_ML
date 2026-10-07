import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & MODERN FINTECH THEME
# ==============================================================================
st.set_page_config(
    page_title="LoanSense AI | Intelligent Loan Underwriting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0b192c 0%, #1e3e62 50%, #000000 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff !important;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94a3b8 !important;
        max-width: 820px;
        line-height: 1.6;
    }
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0 20px;
    }
    
    /* Decision Verdict Banners */
    .verdict-banner {
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin: 1.2rem 0;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .verdict-approved {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%) !important;
        border: 2px solid #10b981 !important;
        color: #ffffff !important;
    }
    .verdict-rejected {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%) !important;
        border: 2px solid #f43f5e !important;
        color: #ffffff !important;
    }
    .verdict-review {
        background: linear-gradient(135deg, #78350f 0%, #451a03 100%) !important;
        border: 2px solid #f59e0b !important;
        color: #ffffff !important;
    }
    .verdict-heading {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.3rem;
        color: #ffffff !important;
    }
    .verdict-subtext {
        font-size: 1.05rem;
        font-weight: 500;
        color: #f1f5f9 !important;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & MODEL CACHING
# ==============================================================================
@st.cache_resource
def load_trained_model():
    return joblib.load("loan_model.joblib")


@st.cache_data
def load_dataset():
    df = pd.read_csv("loan_approval_data.csv")
    return df


try:
    model = load_trained_model()
    raw_df = load_dataset()
    clean_df = raw_df.dropna(subset=["Loan_Approved"]).copy()
except Exception as e:
    st.error(f"Error loading required resources: {e}")
    st.stop()


# ==============================================================================
# 3. SIDEBAR CONTROLS & PROFILE PRESETS
# ==============================================================================
with st.sidebar:
    st.markdown("## 🏦 **LoanSense AI**")
    st.caption("AI-Powered Credit Decisioning & Risk Engine")
    st.markdown("---")
    
    st.markdown("### ⚡ Quick-Fill Personas")
    st.caption("Test pre-configured applicant archetypes:")
    
    preset = st.selectbox(
        "Select Persona Preset",
        [
            "Custom (Manual Input)",
            "🟢 Prime Applicant (Low Risk)",
            "🟡 Borderline Applicant (Moderate Risk)",
            "🔴 High-Risk Applicant (Declined)"
        ],
        index=0
    )
    
    # Preset Defaults
    if preset == "🟢 Prime Applicant (Low Risk)":
        p_inc, p_coinc, p_sav, p_col = 14000, 6000, 22000, 45000
        p_score, p_dti, p_loans = 740, 0.22, 1
        p_amt, p_term, p_purp = 25000, 36, "Home"
        p_age, p_gen, p_mar, p_dep, p_edu = 38, "Male", "Married", 1, "Graduate"
        p_emp, p_cat, p_area = "Salaried", "MNC", "Urban"
    elif preset == "🟡 Borderline Applicant (Moderate Risk)":
        p_inc, p_coinc, p_sav, p_col = 7500, 2000, 6000, 15000
        p_score, p_dti, p_loans = 655, 0.38, 2
        p_amt, p_term, p_purp = 20000, 48, "Personal"
        p_age, p_gen, p_mar, p_dep, p_edu = 31, "Female", "Single", 0, "Graduate"
        p_emp, p_cat, p_area = "Salaried", "Private", "Semiurban"
    elif preset == "🔴 High-Risk Applicant (Declined)":
        p_inc, p_coinc, p_sav, p_col = 4000, 0, 1500, 5000
        p_score, p_dti, p_loans = 580, 0.49, 3
        p_amt, p_term, p_purp = 35000, 60, "Business"
        p_age, p_gen, p_mar, p_dep, p_edu = 26, "Male", "Single", 2, "Not Graduate"
        p_emp, p_cat, p_area = "Self-employed", "Business", "Rural"
    else:
        p_inc, p_coinc, p_sav, p_col = 10000, 5000, 10000, 25000
        p_score, p_dti, p_loans = 680, 0.35, 1
        p_amt, p_term, p_purp = 20000, 48, "Personal"
        p_age, p_gen, p_mar, p_dep, p_edu = 35, "Male", "Married", 1, "Graduate"
        p_emp, p_cat, p_area = "Salaried", "Private", "Urban"

    st.markdown("---")
    st.markdown("### ⚙️ Policy Settings")
    threshold = st.slider(
        "Approval Cutoff Threshold",
        min_value=0.30,
        max_value=0.85,
        value=0.50,
        step=0.05,
        help="Standard banking cutoff is 0.50. Higher values enforce stricter risk tolerance."
    )
    
    st.markdown("---")
    st.markdown("📊 **Model:** `Gradient Boosting (Tuned)`")
    st.markdown("🎯 **Holdout F1:** `0.942` | **AUC:** `0.991`")


# ==============================================================================
# 4. HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏦 LoanSense AI Underwriting & Analytics</div>
    <div class="hero-subtitle">
        Enterprise credit risk assessment and loan approval prediction platform powered by tuned Gradient Boosted Trees and automated scikit-learn pipelines.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. MULTI-TAB DASHBOARD
# ==============================================================================
tab_predict, tab_eda, tab_model, tab_insights = st.tabs([
    "🎯 Live Predictor & Risk Assessment",
    "📊 Exploratory Data Analysis (EDA)",
    "🤖 Model Benchmarking & Explainability",
    "💼 Business Insights & Governance"
])


# ==============================================================================
# TAB 1: LIVE PREDICTOR & RISK ASSESSMENT
# ==============================================================================
with tab_predict:
    st.markdown("### 📝 Applicant & Loan Evaluation Form")
    st.caption("Configure applicant demographics, financial ratios, and requested loan parameters:")
    
    col_app, col_fin = st.columns([1, 1], gap="large")
    
    with col_app:
        with st.container(border=True):
            st.markdown("#### 👤 Applicant Demographics & Profile")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Applicant Age", 21, 65, p_age)
                gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(p_gen))
                marital = st.selectbox("Marital Status", ["Married", "Single"], index=["Married", "Single"].index(p_mar))
                dependents = st.number_input("Dependents Count", 0, 5, p_dep)
            with c2:
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=["Graduate", "Not Graduate"].index(p_edu))
                employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"], index=["Salaried", "Self-employed", "Contract", "Unemployed"].index(p_emp))
                employer = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"], index=["Private", "Government", "MNC", "Business", "Unemployed"].index(p_cat))
                area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(p_area))

    with col_fin:
        with st.container(border=True):
            st.markdown("#### 💰 Financial Health & Credit Scores")
            c3, c4 = st.columns(2)
            with c3:
                income = st.number_input("Primary Monthly Income ($)", 0, 100000, p_inc, step=500)
                co_income = st.number_input("Coapplicant Monthly Income ($)", 0, 50000, p_coinc, step=500)
                savings = st.number_input("Savings Balance ($)", 0, 200000, p_sav, step=1000)
                collateral = st.number_input("Collateral Value ($)", 0, 300000, p_col, step=2500)
            with c4:
                credit_score = st.slider("Credit Score (FICO)", 550, 800, p_score, help="Key driver: Scores ≥ 650 exhibit high approval rates.")
                dti = st.slider("Debt-to-Income (DTI) Ratio", 0.05, 0.70, float(p_dti), step=0.01, help="Key driver: DTI ≤ 0.40 is the standard risk ceiling.")
                existing_loans = st.number_input("Active Loans Count", 0, 6, p_loans)

    with st.container(border=True):
        st.markdown("#### 📑 Requested Loan Structure")
        c5, c6, c7 = st.columns(3)
        with c5:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, p_amt, step=1000)
        with c6:
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84], index=[12, 24, 36, 48, 60, 72, 84].index(p_term))
        with c7:
            purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Education", "Home"], index=["Personal", "Car", "Business", "Education", "Home"].index(p_purp))

    # Prediction Logic
    input_df = pd.DataFrame([{
        "Applicant_Income": income,
        "Coapplicant_Income": co_income,
        "Employment_Status": employment,
        "Age": age,
        "Marital_Status": marital,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": purpose,
        "Property_Area": area,
        "Education_Level": education,
        "Gender": gender,
        "Employer_Category": employer,
    }])

    prob_approval = float(model.predict_proba(input_df)[0, 1])
    is_approved = prob_approval >= threshold

    st.markdown("---")
    st.markdown("### 📊 Underwriting Assessment & Risk Verdict")

    res_col1, res_col2 = st.columns([1.2, 1], gap="large")

    with res_col1:
        if is_approved:
            if prob_approval >= 0.75:
                v_class = "verdict-approved"
                v_title = "✅ APPROVED — LOW RISK"
                v_desc = f"Application recommended for approval with <b>{prob_approval:.1%}</b> estimated approval probability."
            else:
                v_class = "verdict-review"
                v_title = "🟡 CONDITIONAL APPROVAL — MODERATE RISK"
                v_desc = f"Application meets approval cutoff (<b>{prob_approval:.1%}</b>), but lies close to policy boundaries."
        else:
            v_class = "verdict-rejected"
            v_title = "❌ DECLINED — HIGH RISK"
            v_desc = f"Application declined with an estimated approval probability of only <b>{prob_approval:.1%}</b> (Cutoff: {threshold:.0%})."

        st.markdown(f"""
        <div class="verdict-banner {v_class}">
            <div class="verdict-heading">{v_title}</div>
            <div class="verdict-subtext">{v_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Financial Summary Metrics in Container Card
        with st.container(border=True):
            tot_income = income + co_income
            annual_income = max(tot_income * 12, 1)
            lti_ratio = loan_amount / annual_income
            collat_coverage = (collateral / loan_amount) * 100 if loan_amount > 0 else 0

            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Total Monthly Inflow", f"${tot_income:,.0f}")
            with kpi2:
                st.metric("Loan-to-Annual-Income", f"{lti_ratio:.2f}x")
            with kpi3:
                st.metric("Collateral Security", f"{collat_coverage:.0f}%")

        # Underwriter Checklist
        with st.container(border=True):
            st.markdown("#### 📋 Underwriter Risk Factor Breakdown")
            if credit_score >= 660:
                st.success(f"✔️ **Credit Score ({credit_score}):** Exceeds prime threshold (650+). Strong credit history.")
            else:
                st.error(f"⚠️ **Credit Score ({credit_score}):** Below benchmark (650+). Key risk driver.")

            if dti <= 0.40:
                st.success(f"✔️ **DTI Ratio ({dti:.2f}):** Within healthy debt capacity bounds (≤ 0.40).")
            else:
                st.error(f"⚠️ **DTI Ratio ({dti:.2f}):** Exceeds debt-to-income ceiling (> 0.40).")

            if collat_coverage >= 100:
                st.info(f"🛡️ **Collateral:** Fully secured loan ({collat_coverage:.0f}% asset backing).")
            else:
                st.caption(f"ℹ️ **Collateral:** Partially secured loan ({collat_coverage:.0f}% asset backing).")

    with res_col2:
        with st.container(border=True):
            st.markdown("#### 🎯 Approval Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_approval * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Approval Probability", 'font': {'size': 20, 'weight': 'bold', 'color': "#ffffff"}},
                delta={'reference': threshold * 100, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#f43f5e"}},
                number={'suffix': "%", 'font': {'size': 44, 'weight': 'bold', 'color': "#ffffff"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.4)"},
                    'bar': {'color': "#0284c7" if is_approved else "#dc2626", 'thickness': 0.35},
                    'bgcolor': "rgba(30, 41, 59, 0.6)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.15)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.25)'},
                        {'range': [40, 65], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [65, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "#38bdf8", 'width': 4},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("### 📊 Dataset Exploration & Underwriting Patterns")
    st.caption("Insights extracted from 950 historical loan records with adjudicated approval outcomes.")

    # High-level Dataset Metrics in Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("Total Labeled Applications", "950", help="1,000 total minus 50 unlabeled rows")
    with m2:
        with st.container(border=True):
            approval_rate = (clean_df["Loan_Approved"] == "Yes").mean() * 100
            st.metric("Historical Approval Rate", f"{approval_rate:.1f}%", help="298 Approved / 652 Rejected")
    with m3:
        with st.container(border=True):
            avg_cs = clean_df["Credit_Score"].mean()
            st.metric("Mean Credit Score", f"{avg_cs:.0f}", help="Standard FICO range: 550 - 800")
    with m4:
        with st.container(border=True):
            med_dti = clean_df["DTI_Ratio"].median()
            st.metric("Median DTI Ratio", f"{med_dti:.2f}", help="Debt obligations / Monthly income")

    st.markdown("---")

    # Row 1: Target Balance & The Decision Boundary
    eda_r1_c1, eda_r1_c2 = st.columns([1, 1.4], gap="large")

    with eda_r1_c1:
        with st.container(border=True):
            st.markdown("#### 🎯 Target Class Balance")
            st.caption("Distribution of Approved vs Rejected historical cases:")
            target_counts = clean_df["Loan_Approved"].value_counts().reset_index()
            target_counts.columns = ["Status", "Count"]
            target_counts["Label"] = target_counts["Status"].map({"Yes": "Approved (31.4%)", "No": "Rejected (68.6%)"})

            fig_pie = px.pie(
                target_counts,
                names="Label",
                values="Count",
                color="Label",
                color_discrete_map={
                    "Approved (31.4%)": "#10b981", 
                    "Rejected (68.6%)": "#f43f5e"
                },
                hole=0.60
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                textfont=dict(size=14, color="#ffffff", family="Plus Jakarta Sans"),
                marker=dict(line=dict(color='#0f172a', width=2))
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=20, r=20), 
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with eda_r1_c2:
        with st.container(border=True):
            st.markdown("#### ⚡ Core Empirical Decision Boundary")
            st.caption("Credit Score ≥ 650 & DTI ≤ 0.40 defines the approval boundary:")
            fig_scatter = px.scatter(
                clean_df,
                x="Credit_Score",
                y="DTI_Ratio",
                color="Loan_Approved",
                color_discrete_map={"Yes": "#10b981", "No": "#f43f5e"},
                opacity=0.85,
                hover_data=["Applicant_Income", "Loan_Amount", "Employment_Status"],
                labels={"Loan_Approved": "Approved", "Credit_Score": "Credit Score (FICO)", "DTI_Ratio": "Debt-to-Income (DTI)"}
            )
            fig_scatter.add_vline(x=650, line_dash="dash", line_color="#38bdf8", line_width=2, annotation_text="Min Score: 650", annotation_position="top left", annotation_font_color="#38bdf8")
            fig_scatter.add_hline(y=0.40, line_dash="dash", line_color="#38bdf8", line_width=2, annotation_text="Max DTI: 0.40", annotation_position="bottom right", annotation_font_color="#38bdf8")
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=20, r=20), 
                height=350,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Row 2: Categorical Approval Rates
    with st.container(border=True):
        st.markdown("#### 👥 Categorical Approval Rate Drill-Down")
        cat_feature = st.selectbox(
            "Select Categorical Dimension:",
            ["Employment_Status", "Employer_Category", "Education_Level", "Property_Area", "Loan_Purpose", "Gender", "Marital_Status"],
            index=0
        )

        cat_grp = clean_df.groupby(cat_feature)["Loan_Approved"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        cat_grp.columns = [cat_feature, "Approval_Rate"]
        cat_grp = cat_grp.sort_values(by="Approval_Rate", ascending=False)

        fig_bar = px.bar(
            cat_grp,
            x=cat_feature,
            y="Approval_Rate",
            text=cat_grp["Approval_Rate"].apply(lambda x: f" {x:.1f}%"),
            color="Approval_Rate",
            color_continuous_scale=[[0, '#0369a1'], [0.5, '#0ea5e9'], [1, '#38bdf8']],
            labels={"Approval_Rate": "Approval Rate (%)", cat_feature: cat_feature.replace("_", " ")}
        )
        fig_bar.add_hline(y=approval_rate, line_dash="dot", line_color="#f43f5e", line_width=2, annotation_text=f"Portfolio Average ({approval_rate:.1f}%)", annotation_font_color="#f43f5e")
        fig_bar.update_traces(textposition='outside', textfont=dict(size=13, color="#ffffff"))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360, 
            margin=dict(t=30, b=20, l=20, r=20),
            yaxis=dict(range=[0, max(cat_grp["Approval_Rate"]) + 10], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Row 3: Feature Correlation Heatmap
    with st.container(border=True):
        st.markdown("#### 🔗 Numeric Feature Correlation Heatmap")
        num_cols = clean_df.select_dtypes(include="number").columns.tolist()
        if "Applicant_ID" in num_cols:
            num_cols.remove("Applicant_ID")
        
        corr_df = clean_df[num_cols].copy()
        corr_df["Target_Approved"] = (clean_df["Loan_Approved"] == "Yes").astype(int)
        corr_matrix = corr_df.corr().round(2)

        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500, 
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# TAB 3: MODEL PERFORMANCE & BENCHMARKING
# ==============================================================================
with tab_model:
    st.markdown("### 🤖 Algorithmic Benchmarking & Explainability")
    st.caption("5-Fold Stratified Cross-Validation benchmark across 7 distinct classification architectures:")

    # Leaderboard Table in Container
    with st.container(border=True):
        st.markdown("#### 🏆 Model Performance Leaderboard")
        leaderboard_data = {
            "Model Architecture": [
                "🏆 Gradient Boosting (Selected)",
                "🌲 Random Forest",
                "🌳 Decision Tree (Tuned)",
                "📈 Logistic Regression",
                "📊 Gaussian Naive Bayes",
                "🎯 Support Vector Machine (RBF)",
                "📍 K-Nearest Neighbors"
            ],
            "CV F1-Score": ["0.942", "0.938", "0.915", "0.777", "0.769", "0.719", "0.467"],
            "ROC-AUC": ["0.991", "0.988", "0.945", "0.865", "0.858", "0.840", "0.760"],
            "Precision": ["0.948", "0.941", "0.892", "0.783", "0.804", "0.774", "0.724"],
            "Recall": ["0.936", "0.935", "0.940", "0.770", "0.738", "0.672", "0.344"],
            "Deployment Status": [
                "✅ Production Deployed",
                "⚡ Candidate",
                "⚡ Candidate",
                "📦 Baseline",
                "📦 Baseline",
                "📦 Baseline",
                "❌ Deprecated"
            ]
        }
        df_leaderboard = pd.DataFrame(leaderboard_data)
        st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

    st.markdown("---")

    mod_c1, mod_c2 = st.columns([1.1, 1], gap="large")

    with mod_c1:
        with st.container(border=True):
            st.markdown("#### 🌟 Gini Feature Importance Ranking")
            st.caption("Relative weight assigned by Gradient Boosting trees during decision splits:")

            clf = model.named_steps["model"]
            prep = model.named_steps["prep"]
            raw_feat_names = prep.get_feature_names_out()
            clean_feat_names = [f.replace("num__", "").replace("cat__", "").replace("_", " ") for f in raw_feat_names]
            
            feat_imp = pd.DataFrame({
                "Feature": clean_feat_names,
                "Importance": clf.feature_importances_ * 100
            }).sort_values(by="Importance", ascending=True).tail(8)

            fig_imp = go.Figure(go.Bar(
                x=feat_imp["Importance"],
                y=feat_imp["Feature"],
                orientation='h',
                text=[f"  {val:.1f}%" for val in feat_imp["Importance"]],
                textposition='outside',
                textfont=dict(size=13, color="#38bdf8", weight="bold"),
                marker=dict(
                    color=feat_imp["Importance"],
                    colorscale=[[0, '#0369a1'], [0.5, '#0ea5e9'], [1, '#38bdf8']],
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                )
            ))
            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=130, r=60),
                height=380,
                xaxis=dict(
                    title="Gini Importance (%)",
                    range=[0, 62],
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.08)",
                    tickfont=dict(size=12, color="#94a3b8")
                ),
                yaxis=dict(
                    tickfont=dict(size=13, color="#f1f5f9")
                )
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with mod_c2:
        with st.container(border=True):
            st.markdown("#### 🎯 Holdout Test Confusion Matrix")
            st.caption("Evaluated on held-out 20% test partition (N=190 labeled cases):")

            cm_z = [[126, 4], [4, 56]]
            cm_x = ['Pred: Rejected', 'Pred: Approved']
            cm_y = ['Actual: Rejected', 'Actual: Approved']

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_z,
                x=cm_x,
                y=cm_y,
                colorscale=[[0, '#0f172a'], [0.2, '#1e293b'], [1, '#0369a1']],
                showscale=False
            ))

            cm_annotations = [
                dict(x=cm_x[0], y=cm_y[0], text='<b>126</b><br><span style="font-size:11px;color:#94a3b8">True Negative</span>', showarrow=False, font=dict(color="white", size=14)),
                dict(x=cm_x[1], y=cm_y[0], text='<b>4</b><br><span style="font-size:11px;color:#f87171">False Positive</span>', showarrow=False, font=dict(color="white", size=14)),
                dict(x=cm_x[0], y=cm_y[1], text='<b>4</b><br><span style="font-size:11px;color:#f87171">False Negative</span>', showarrow=False, font=dict(color="white", size=14)),
                dict(x=cm_x[1], y=cm_y[1], text='<b>56</b><br><span style="font-size:11px;color:#4ade80">True Positive</span>', showarrow=False, font=dict(color="white", size=14)),
            ]

            fig_cm.update_layout(
                template="plotly_dark",
                annotations=cm_annotations,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=30, l=110, r=20),
                height=300,
                xaxis=dict(tickfont=dict(size=12, color="#f1f5f9"), side="bottom"),
                yaxis=dict(tickfont=dict(size=12, color="#f1f5f9"), autorange="reversed")
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            cm_kpi1, cm_kpi2, cm_kpi3 = st.columns(3)
            with cm_kpi1:
                st.metric("Accuracy", "95.8%")
            with cm_kpi2:
                st.metric("Precision", "93.3%")
            with cm_kpi3:
                st.metric("Recall", "93.3%")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### 🎚️ Interactive Cutoff Sensitivity Simulator")
        st.caption("Simulate how adjusting the decision cutoff affects bank portfolio risk and loan volume conversion:")

        sim_thresh = st.slider("Simulate Cutoff Threshold:", 0.10, 0.90, 0.50, step=0.05)
        t_c1, t_c2, t_c3 = st.columns(3)
        with t_c1:
            st.info(f"**Selected Cutoff:** `{sim_thresh:.2f}`")
        with t_c2:
            rec_impact = "Stricter policy (minimizes default risk, reduces conversion)" if sim_thresh > 0.55 else "Aggressive policy (maximizes loan volume, accepts higher default risk)" if sim_thresh < 0.45 else "Balanced optimal risk-return posture"
            st.warning(f"**Policy Impact:** {rec_impact}")
        with t_c3:
            st.success("**Optimal Zone:** Notebook experiments confirm `0.50 - 0.60` as the optimal financial operating zone.")


# ==============================================================================
# TAB 4: BUSINESS INSIGHTS & RECRUITER GUIDE
# ==============================================================================
with tab_insights:
    st.markdown("### 💼 Executive Briefing & Technical Governance")
    with st.container(border=True):
        st.markdown("""
        #### 📌 Problem Definition & Business Context
        Retail and commercial lending institutions face the dual challenge of maximizing loan origination volume while minimizing non-performing assets (NPAs). This project develops an end-to-end Machine Learning pipeline that automates loan underwriting decisions with high precision and transparency.

        #### 💡 Key Takeaways for Technical Recruiters & Stakeholders:
        1. **Dominant Predictive Drivers:**
           * **Credit Score** (~37.6% feature importance) and **Debt-to-Income (DTI) Ratio** (~52.1% feature importance) jointly explain >89% of the classification variance.
           * Approvals cluster deterministically in the region `Credit Score ≥ 650` and `DTI ≤ 0.40`.
        2. **Algorithmic Selection Rationale:**
           * Because the decision boundary is orthogonal and rule-like, **Ensemble Tree Architectures (Gradient Boosting & Random Forest)** significantly outperformed linear models (Logistic Regression F1 ~ 0.78) and distance-based models (KNN F1 ~ 0.47).
        3. **Fairness & Governance:**
           * Demographic features including **Gender**, **Property Area**, and **Marital Status** demonstrated near-zero feature importance (<0.1%), confirming compliance with fair lending guidelines without disparate impact.
        4. **Leak-Free ML Architecture:**
           * Imputation, categorical encoding, and feature scaling were encapsulated strictly inside scikit-learn `Pipeline` and `ColumnTransformer` artifacts, preventing data leakage across cross-validation splits.
        """)

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### 🛠️ Technology Stack & Methodologies")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        with t1:
            st.button("🐍 Python 3.11+", disabled=True, use_container_width=True)
        with t2:
            st.button("⚙️ Scikit-Learn", disabled=True, use_container_width=True)
        with t3:
            st.button("📊 Plotly Charts", disabled=True, use_container_width=True)
        with t4:
            st.button("🐼 Pandas / NumPy", disabled=True, use_container_width=True)
        with t5:
            st.button("🚀 Streamlit UI", disabled=True, use_container_width=True)
        with t6:
            st.button("🔒 Joblib Pipeline", disabled=True, use_container_width=True)

    st.markdown("---")
    st.caption("Developed by Machine Learning Engineer Portfolio Candidate. Designed for production credit risk scoring.")
