"""
Kaycore Global Partnership Analytics Pro
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import sys
import os


# Temporary - remove after testing
st.cache_resource.clear()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Allow imports from src/
sys.path.append('../')
from src.ml.partnership_models import PartnershipMLModels


# PAGE CONFIG
st.set_page_config(
    page_title="Kaycore Global Partnerships",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS
st.markdown("""
<style>
.main-header {
    font-size: 42px;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 10px;
}
.sub-header {
    font-size: 18px;
    color: #666;
    text-align: center;
    margin-bottom: 30px;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.success-box {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/partners_with_clusters.csv')
    market_sizing = pd.read_csv('data/processed/market_sizing.csv', index_col=0, header=None)[1].to_dict()
    return df, market_sizing


# LOAD MODELS
@st.cache_resource
def load_models():
    rf_path = 'data/models/rf_model.pkl'
    scaler_path = 'data/models/scaler.pkl'
    
    rf_model = None
    scaler = None
    kmeans = None  # you don't save/load kmeans yet, so keep as None
    
    try:
        if not os.path.exists(rf_path):
            st.error(f"RF model missing: {rf_path}")
        else:
            rf_model = joblib.load(rf_path)
            st.success(f"RF model loaded from {rf_path}")
        
        if not os.path.exists(scaler_path):
            st.error(f"Scaler missing: {scaler_path}")
        else:
            scaler = joblib.load(scaler_path)
            st.success(f"Scaler loaded from {scaler_path}")
            
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
    
    return rf_model, scaler, kmeans


# LOAD MODELS
@st.cache_resource
def load_models():
    rf_model = None
    scaler = None
    kmeans = None  # still None since you don't save/load it yet
    
    try:
        rf_model = joblib.load('data/models/rf_model.pkl')
        scaler = joblib.load('data/models/scaler.pkl')
    except Exception as e:
        st.error(f"Failed to load ML models: {str(e)}")
    
    return rf_model, scaler, kmeans


# Load everything
df, market_sizing = load_data()
rf_model, scaler, kmeans = load_models()  # ← this calls the function

# 🔧 GLOBAL FIX: Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

city_col = next(
    (c for c in ['location_city', 'location_city_x', 'location_city_y'] if c in df.columns),
    None
)

df.rename(columns={city_col: "location_city"}, inplace=True)
city_col = "location_city"


# HEADER
st.markdown('<p class="main-header">Kaycore Global Partnership Analytics Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Strategic Partner Discovery & Revenue Optimization Across 4 Countries</p>', unsafe_allow_html=True)


# SIDEBAR FILTERS
st.sidebar.title("Filters")
st.sidebar.markdown("---")

countries = st.sidebar.multiselect(
    "Select Countries",
    options=['SA', 'UK', 'US', 'AU'],
    default=['SA', 'UK', 'US', 'AU']
)

min_fit, max_fit = st.sidebar.slider(
    "Kaycore Fit Score Range",
    0.0, 10.0, (5.0, 10.0), 0.5
)

min_rev, max_rev = st.sidebar.slider(
    "Revenue Range (USD)",
    0, 5000000, (100000, 5000000), 100000,
    format="$%d"
)

priority = st.sidebar.multiselect(
    "Partnership Priority",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium']
)

wp_only = st.sidebar.checkbox("WordPress Specialists Only", value=False)

filtered_df = df[
    (df['country'].isin(countries)) &
    (df['kaycore_fit_score'] >= min_fit) &
    (df['kaycore_fit_score'] <= max_fit) &
    (df['revenue_usd'] >= min_rev) &
    (df['revenue_usd'] <= max_rev) &
    (df['partnership_priority'].isin(priority))
]

if wp_only:
    filtered_df = filtered_df[filtered_df['is_wp_specialist'] == True]

if len(filtered_df) == 0:
    st.warning("No partners match your filters. Try adjusting your selections.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} partners**")


# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview Dashboard",
    "Partner Discovery",
    "ML Predictions",
    "Revenue Simulator",
    "Proposal Generator"
])


# TAB 1: Overview Dashboard
with tab1:

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Partners",
            f"{len(filtered_df)}",
            delta=f"{len(filtered_df)/len(df)*100:.0f}% of total"
        )

    with col2:
        st.metric(
            "Total Revenue Potential",
            f"${filtered_df['revenue_usd'].sum()/1e6:.1f}M",
            delta=f"{filtered_df['revenue_usd'].sum()/df['revenue_usd'].sum()*100:.0f}% of market"
        )

    with col3:
        st.metric(
            "Avg Fit Score",
            f"{filtered_df['kaycore_fit_score'].mean():.1f}/10",
            delta=f"{(filtered_df['kaycore_fit_score'].mean() - df['kaycore_fit_score'].mean()):.1f}"
        )

    with col4:
        high_priority = len(filtered_df[filtered_df['partnership_priority'] == 'High'])
        st.metric(
            "High Priority",
            f"{high_priority}",
            delta=f"{high_priority/len(filtered_df)*100:.0f}%"
        )

    with col5:
        wp_specialists = filtered_df['is_wp_specialist'].sum()
        st.metric(
            "WP Specialists",
            f"{wp_specialists}",
            delta=f"{wp_specialists/len(filtered_df)*100:.0f}%"
        )

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Partners by Country")
        country_counts = filtered_df['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        fig1 = px.bar(
            country_counts,
            x='country',
            y='count',
            color='country',
            title="Geographic Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Revenue Potential by Country")
        revenue_by_country = filtered_df.groupby('country')['revenue_usd'].sum().reset_index()
        fig2 = px.pie(
            revenue_by_country,
            values='revenue_usd',
            names='country',
            title="Revenue Distribution",
            hole=0.4
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Partnership Fit vs Revenue Potential")

    # Remove heavy text fields before plotting
    plot_df = filtered_df.drop(columns=['services'], errors='ignore')

    # FIX: Limit to top 1200 points by fit score to reduce data size
    plot_df = plot_df.nlargest(1200, 'kaycore_fit_score').copy()

    fig3 = px.scatter(
        plot_df,
        x='kaycore_fit_score',
        y='revenue_usd',
        color='country',
        size='employees',
        hover_name='name',
        hover_data=['location_city', 'clutch_rating'],  # FIX: Reduced hover_data to minimize payload
        title="Partner Portfolio Analysis"
    )

    fig3.update_yaxes(type='log')
    st.plotly_chart(
    fig3,
    use_container_width=True,          # this one is still allowed as top-level kwarg
    config={
        "responsive": True,
        "displayModeBar": True,        # optional: shows toolbar
        # "render": "browser"          # ← NOT valid in config (see below)
    }
)

    st.markdown("---")
    st.subheader("Market Sizing Analysis")

    col_tam, col_sam, col_som = st.columns(3)

    with col_tam:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### TAM")
        st.markdown(f"## ${float(market_sizing.get('tam_usd', 0))/1e6:.1f}M")
        st.markdown(f"{int(market_sizing.get('tam_partners', 0))} partners")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sam:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### SAM")
        st.markdown(f"## ${float(market_sizing.get('sam_usd', 0))/1e6:.1f}M")
        st.markdown(f"{int(market_sizing.get('sam_partners', 0))} partners")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_som:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### SOM")
        st.markdown(f"## ${float(market_sizing.get('som_usd', 0))/1e6:.1f}M")
        st.markdown(f"{int(market_sizing.get('som_partners', 0))} partners")
        st.markdown('</div>', unsafe_allow_html=True)


# TAB 2: Partner Discovery
with tab2:

    st.subheader("Top Partnership Candidates")

    sort_col = st.selectbox(
        "Sort by",
        options=['kaycore_fit_score', 'revenue_usd', 'clutch_rating', 'employees'],
        format_func=lambda x: {
            'kaycore_fit_score': 'Fit Score',
            'revenue_usd': 'Revenue',
            'clutch_rating': 'Rating',
            'employees': 'Company Size'
        }[x]
    )

    top_partners = filtered_df.nlargest(50, sort_col)

    display_df = top_partners[[ 
        'name', 
        'country', 
        'location_city', 
        'website', 
        'employees', 
        'revenue_usd', 
        'kaycore_fit_score',
        'clutch_rating', 
        'partnership_priority', 
        'is_wp_specialist' 
     ]].copy()

    display_df['revenue_usd'] = display_df['revenue_usd'].apply(lambda x: f"${x:,.0f}")

    display_df.columns = [
        'Agency Name',
        'Country',
        'City',
        'Website',
        'Employees',
        'Est. Revenue',
        'Fit Score',
        'Rating',
        'Priority',
        'WP Specialist'
    ]

    st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)

    csv = top_partners.to_csv(index=False)
    st.download_button(
        label="Download Top 50 Partners (CSV)",
        data=csv,
        file_name=f"kaycore_top_partners_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# TAB 3: ML Predictions
with tab3:

    st.subheader("AI-Powered Partner Scoring")
    st.markdown("""
    Enter a new prospect's details to get instant ML predictions for:
    - **Success Probability**
    - **Revenue Forecast**
    - **Priority Recommendation**
    """)

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            new_name = st.text_input("Agency Name", "Example Agency")
            new_country = st.selectbox("Country", ['SA', 'UK', 'US', 'AU'])
            new_employees = st.number_input("Employees", 1, 1000, 50)

        with col2:
            new_rating = st.slider("Clutch Rating", 0.0, 5.0, 4.5, 0.1)
            new_project_size = st.number_input("Min Project Size ($)", 0, 100000, 10000, 1000)
            new_wp = st.checkbox("WordPress Specialist", value=False)

        with col3:
            new_services = st.multiselect(
                "Services Offered",
                ['Web Development', 'WordPress', 'SEO', 'Hosting', 'Maintenance'],
                default=['Web Development']
            )

        predict_button = st.form_submit_button("Predict Partnership Potential")

    if predict_button:

        ml_engine = PartnershipMLModels()
        ml_engine.rf_model = rf_model
        ml_engine.scaler = scaler
        ml_engine.kmeans_model = kmeans

        revenue_estimate = new_employees * 100000 * (new_rating / 5.0)

        fit_score = 0
        if new_wp:
            fit_score += 3
        if 25 <= new_employees <= 100:
            fit_score += 2
        if new_rating >= 4.5:
            fit_score += 2
        fit_score += len(new_services) * 0.5
        fit_score = min(fit_score, 10.0)

        new_partner_data = {
            'employees': new_employees,
            'revenue_usd': revenue_estimate,
            'clutch_rating': new_rating,
            'min_project_size_usd': new_project_size,
            'is_wp_specialist': int(new_wp),
            'is_enterprise': int(new_employees > 50),
            'kaycore_fit_score': fit_score,
            'country': new_country
        }

        predictions = ml_engine.predict_new_partner(new_partner_data)

        st.markdown("---")
        st.success("Prediction Complete!")

        col_pred1, col_pred2, col_pred3 = st.columns(3)

        with col_pred1:
            success_prob = predictions['success_probability'] * 100
            st.metric(
                "Success Probability",
                f"{success_prob:.1f}%",
                delta="High Confidence" if success_prob > 70 else "Medium Confidence"
            )

        with col_pred2:
            pred_revenue = predictions['predicted_revenue_usd']
            st.metric(
                "Predicted Revenue",
                f"${pred_revenue:,.0f}",
                delta="Year 1 Estimate"
            )

        with col_pred3:
            if success_prob >= 70:
                priority = "High Priority"
            elif success_prob >= 50:
                priority = "Medium Priority"
            else:
                priority = "Low Priority"

            st.metric("Recommendation", priority)

        st.markdown("---")
        st.subheader("Partnership Insights")

        st.markdown(f"""
        <div class="success-box">
        <strong>Analysis for {new_name}</strong><br><br>
        <strong>Strengths:</strong><br>
        - Fit Score: {fit_score:.1f}/10<br>
        - Employees: {new_employees}<br>
        - Rating: {new_rating}/5.0<br>
        - WordPress Specialist: {"Yes" if new_wp else "No"}<br><br>

        <strong>Revenue Potential:</strong><br>
        - Year 1: ${pred_revenue:,.0f}<br>
        - Year 2 (30% growth): ${pred_revenue * 1.3:,.0f}<br>
        - 3-Year Total: ${pred_revenue * 3.6:,.0f}<br>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────
# TAB 4: Revenue Simulator
# ────────────────────────────────────────────────
with tab4:
    st.session_state.tab_key = 4
    
    st.subheader("Revenue Growth Simulator")
    st.markdown("Adjust sliders to model scenarios. Charts appear only when toggled below.")

    # Always-visible inputs & quick results
    col1, col2 = st.columns(2)

    with col1:
        partners_y1 = st.slider(
            "Year 1 Partners", 5, 200, 50, 5,
            help="Number of partner agencies in Year 1"
        )
        avg_deal = st.slider(
            "Average Deal Size ($)", 5000, 150000, 50000, 5000,
            format="${:,}"
        )
        recur_rate = st.slider(
            "Recurring Revenue Rate (%)", 40, 100, 70, 5
        )
        growth_y2 = st.slider(
            "Year 2 Partner Growth (%)", 0, 100, 30, 5
        )

    # Calculations (always computed)
    y1_rev = partners_y1 * avg_deal
    y1_rec = y1_rev * (recur_rate / 100)
    partners_y2 = int(partners_y1 * (1 + growth_y2 / 100))
    y2_new = (partners_y2 - partners_y1) * avg_deal
    y2_total = y1_rec + y2_new
    growth_pct = ((y2_total - y1_rev) / y1_rev) * 100 if y1_rev > 0 else 0

    with col2:
        st.metric("Year 1 Revenue", f"${y1_rev:,.0f}")
        st.metric(
            "Year 2 Revenue", 
            f"${y2_total:,.0f}", 
            delta=f"+{growth_pct:.1f}%" if growth_pct != 0 else "0%"
        )
        st.metric("Year 2 Partners", partners_y2)

        if growth_pct >= 50:
            st.success(f"50%+ growth achieved! ({growth_pct:.1f}%)")
        else:
            st.warning(f"Need {50 - growth_pct:.1f}% more to hit 50%")

    # Charts — only render if user checks the box (prevents bleed)
    show_charts = st.checkbox("Show Growth Charts", value=False, key="show_growth_charts_tab4")

    if show_charts:
        st.markdown("---")
        st.subheader("Projected Growth Visuals")

        # Revenue bar
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(
            x=['Year 1', 'Year 2', 'Year 3 (Proj)'],
            y=[y1_rev, y2_total, y2_total * 1.2],
            name='Revenue',
            marker_color='#667eea'
        ))
        fig_rev.update_layout(
            title="Revenue Growth Projection",
            xaxis_title="Year",
            yaxis_title="Revenue (USD)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_rev, use_container_width=True, config={'responsive': True})

        # Partner line
        fig_part = go.Figure()
        fig_part.add_trace(go.Scatter(
            x=['Year 1', 'Year 2', 'Year 3 (Proj)'],
            y=[partners_y1, partners_y2, int(partners_y2 * 1.15)],
            mode='lines+markers',
            name='Partners',
            line=dict(color='#764ba2', width=3)
        ))
        fig_part.update_layout(
            title="Partner Growth Projection",
            xaxis_title="Year",
            yaxis_title="Number of Partners",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_part, use_container_width=True, config={'responsive': True})
    else:
        st.info("Check the box above to view charts.")


# ────────────────────────────────────────────────
# TAB 5: Proposal Generator
# ────────────────────────────────────────────────
with tab5:
    # Force state tracking to help with tab isolation
    if 'last_tab' not in st.session_state:
        st.session_state.last_tab = None

    st.session_state.tab_key = 5
    st.markdown(" ")           # flush
    st.empty()                 # flush

    st.subheader("Automated Partnership Proposal Generator")
    st.markdown("Select a strong candidate to generate a draft outreach proposal.")

    # Broader candidate pool
    proposal_candidates = df[
        (df['kaycore_fit_score'] >= 3.0) &
        (df['partnership_priority'].isin(['High', 'Medium'])) &
        (df['employees'] >= 10)
    ].nlargest(50, 'kaycore_fit_score')

    if proposal_candidates.empty:
        st.warning("No strong candidates available in the dataset.")
    else:
        st.caption(f"Showing {len(proposal_candidates)} top candidates")

        selected_name = st.selectbox(
            "Select Partner",
            options=proposal_candidates['name'].tolist(),
            index=0,
            key="proposal_partner_select"
        )

        if selected_name:
            partner = proposal_candidates[proposal_candidates['name'] == selected_name].iloc[0]

            col_left, col_right = st.columns([3, 1])

            with col_left:
                cluster_label = f"Cluster {int(partner['cluster'])}"
                if partner['cluster'] == 3:
                    cluster_label += " (Mobile-heavy)"

                wp_status = "Yes (in services)" if partner.get('services_wp', False) else \
                            "Yes (tagged)" if partner.get('is_wp_specialist', False) else "No"

                revenue_formatted = f"${partner['revenue_usd']:,.0f}" if pd.notna(partner['revenue_usd']) else "N/A"

                st.markdown(f"""
                ### Draft Outreach Proposal — {partner['name']}

                **Date:** {datetime.now().strftime('%B %d, %Y')}  
                **To:** {partner['name']}  
                **Location:** {partner.get('location_city', 'N/A')}, {partner['country']}

                ---

                **Subject:** Strategic Partnership Opportunity – Kaycore Creatives × {partner['name']}

                Dear {partner['name']} Team,

                Our analytics platform has identified **{partner['name']}** as a high-potential partner:

                - **Fit Score**: {partner['kaycore_fit_score']:.1f}/10  
                - **Est. Revenue**: {revenue_formatted}  
                - **Team Size**: {partner['employees']} employees  
                - **Cluster**: {cluster_label}  
                - **Mobile Focus**: {partner.get('mobile_pct', 0):.0f}%  
                - **WordPress Capabilities**: {wp_status}  
                - **Clutch Rating**: {partner['clutch_rating']:.2f}/5.0

                We propose collaborating to deliver white-labeled WordPress security, performance optimization, and maintenance solutions — creating new recurring revenue for both sides.

                **Proposed Terms**  
                - Revenue share: 70% {partner['name']} / 30% Kaycore  
                - Year 1 joint target: ${partner['revenue_usd'] * 0.10:,.0f} – ${partner['revenue_usd'] * 0.15:,.0f}  
                - Typical deal size: $8,000–$15,000 per client  
                - Expected new clients: {max(5, int((partner['revenue_usd'] * 0.12) / 10000))}

                **Next Steps**  
                1. Schedule 20–30 min discovery call  
                2. Live demo + case studies  
                3. Review draft partnership agreement

                Best regards,  
                Surprise Fakude  
                Director, Kaycore Creatives  
                surprise@kaycorecreatives.com  
                kaycorecreatives.com
                """, unsafe_allow_html=True)

            with col_right:
                st.markdown("### Quick Snapshot")
                st.metric("Fit Score", f"{partner['kaycore_fit_score']:.1f}/10")
                st.metric("Est. Revenue", revenue_formatted)
                st.metric("Employees", partner['employees'])
                st.metric("Cluster", cluster_label)
                st.metric("Mobile Focus", f"{partner.get('mobile_pct', 0):.0f}%")
                st.metric("WP Capabilities", wp_status)

                st.markdown("---")

                if st.button("Copy Proposal Text"):
                    st.success("Proposal text copied! Paste into email.")

                st.caption("PDF & email features – coming soon")

    # One-time rerun on first visit (modern Streamlit version)
    if st.session_state.last_tab != 5:
        st.session_state.last_tab = 5
        st.rerun()  # ← This is the corrected call


# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
<strong>Kaycore Global Partnership Analytics Pro v1.0</strong><br>
Built by Suraj Raut & Saul Guzman | Data Science Interns 2026<br>
SA | UK | US | AU
</div>
""", unsafe_allow_html=True)
