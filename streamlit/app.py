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
import sys

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
    rf_model = joblib.load('data/models/rf_model.pkl')
    scaler = None
    kmeans = None

    return rf_model, scaler, kmeans


df, market_sizing = load_data()
rf_model, scaler, kmeans = load_models()


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

# ⭐ Add this block to prevent division by zero
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
    fig3 = px.scatter(
        filtered_df,
        x='kaycore_fit_score',
        y='revenue_usd',
        color='country',
        size='employees',
        hover_data=['name', 'clutch_rating'],
        title="Partner Portfolio Analysis"
    )
    fig3.update_yaxes(type='log')
    st.plotly_chart(fig3, use_container_width=True)

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

    # Select only columns that exist
    display_df = top_partners[[
        'name',
        'country',
        'employees',
        'revenue_usd',
        'kaycore_fit_score',
        'clutch_rating',
        'partnership_priority',
        'is_wp_specialist'
    ]].copy()

    # Format revenue
    display_df['revenue_usd'] = display_df['revenue_usd'].apply(lambda x: f"${x:,.0f}")

    # Rename columns
    display_df.columns = [
        'Agency Name',
        'Country',
        'Employees',
        'Est. Revenue',
        'Fit Score',
        'Rating',
        'Priority',
        'WP Specialist'
    ]

    st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)

    # CSV download
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


# TAB 4: Revenue Simulator
with tab4:

    st.subheader("50% Revenue Growth Simulator")
    st.markdown("Model different partnership scenarios to achieve **50% revenue growth**.")

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        num_partners_y1 = st.slider("Partners (Year 1)", 10, 150, 50, 5)
        avg_deal_size = st.slider("Average Deal Size ($)", 10000, 200000, 50000, 5000)
        recurring_rate = st.slider("Recurring Revenue Rate (%)", 50, 100, 70, 5)
        growth_rate_y2 = st.slider("Year 2 Partner Growth (%)", 10, 100, 30, 5)

    with col_sim2:
        y1_revenue = num_partners_y1 * avg_deal_size
        y1_recurring = y1_revenue * (recurring_rate / 100)
        num_partners_y2 = int(num_partners_y1 * (1 + growth_rate_y2 / 100))
        y2_new_revenue = (num_partners_y2 - num_partners_y1) * avg_deal_size
        y2_total = y1_recurring + y2_new_revenue
        growth_achieved = ((y2_total - y1_revenue) / y1_revenue) * 100

        st.metric("Year 1 Revenue", f"${y1_revenue:,.0f}")
        st.metric("Year 2 Revenue", f"${y2_total:,.0f}", delta=f"+{growth_achieved:.1f}%")
        st.metric("Total Partners (Y2)", num_partners_y2)

        if growth_achieved >= 50:
            st.success(f"Target achieved! {growth_achieved:.1f}% growth")
        else:
            st.warning(f"Need {50 - growth_achieved:.1f}% more growth")

    st.markdown("---")

    years = ['Year 1', 'Year 2', 'Year 3 (Projected)']
    revenues = [y1_revenue, y2_total, y2_total * 1.2]
    partners = [num_partners_y1, num_partners_y2, int(num_partners_y2 * 1.15)]

    fig_sim = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue Growth', 'Partner Growth'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )

    fig_sim.add_trace(
        go.Bar(x=years, y=revenues, name='Revenue', marker_color='#667eea'),
        row=1, col=1
    )

    fig_sim.add_trace(
        go.Scatter(x=years, y=partners, mode='lines+markers', name='Partners',
                   line=dict(color='#764ba2', width=3)),
        row=1, col=2
    )

    st.plotly_chart(fig_sim, use_container_width=True)
# TAB 5: Proposal Generator
with tab5:

    st.subheader("Automated Partnership Proposal Generator")
    st.markdown("Select a partner to generate a customized partnership proposal.")

    high_priority_partners = filtered_df[filtered_df['partnership_priority'] == 'High'].nlargest(20, 'kaycore_fit_score')

    selected_partner_name = st.selectbox(
        "Select Partner",
        options=high_priority_partners['name'].tolist()
    )

    if selected_partner_name:

        partner = high_priority_partners[high_priority_partners['name'] == selected_partner_name].iloc[0]

        col_prop1, col_prop2 = st.columns([2, 1])

        with col_prop1:
            st.markdown(f"""
            ### Partnership Proposal for {partner['name']}

            **Date:** {datetime.now().strftime('%B %d, %Y')}  
            **Prepared by:** Kaycore Creatives  

            ---

            #### Executive Summary  
            Kaycore Creatives proposes a strategic partnership with **{partner['name']}** to expand our joint market presence in **{partner['location_city']}, {partner['country']}** and deliver enhanced WordPress security solutions to your client base.

            #### Partnership Fit Analysis  
            - **Kaycore Fit Score:** {partner['kaycore_fit_score']:.1f}/10  
            - **Company Profile:** {partner['employees']} employees  
            - **Estimated Revenue:** ${partner['revenue_usd']:,.0f}  
            - **Rating:** {partner['clutch_rating']}/5.0  
            - **Specialization:** {"WordPress Development" if partner['is_wp_specialist'] else "Web Development"}

            #### Proposed Revenue Model  
            **Year 1 Projections:**  
            - Joint revenue target: ${partner['revenue_usd'] * 0.15:,.0f}  
            - Revenue share: **70% {partner['name']} / 30% Kaycore**  
            - Minimum deal value: $5,000 per client  
            - Expected clients: {int((partner['revenue_usd'] * 0.15) / 5000)}

            #### Joint Offering — SecureShield Pro Partnership  
            - White-label SecureShield for your clients  
            - Co-branded marketing materials  
            - Technical training & support  
            - Sales enablement resources  

            **Service Bundle:**  
            - WordPress Security Audits  
            - Malware Scanning & Removal  
            - Performance Optimization  
            - Ongoing Maintenance Contracts  

            #### Benefits to {partner['name']}  
            - **Revenue Growth:** Additional $50K–$150K in Year 1  
            - **Client Retention:** Enhanced security offering  
            - **Market Differentiation:** Exclusive partnership territory  
            - **Support:** Dedicated partner success manager  

            #### Next Steps  
            1. **Discovery Call:** Schedule 30‑min intro meeting  
            2. **Technical Demo:** SecureShield platform walkthrough  
            3. **Contract Review:** Terms, pricing, territory  
            4. **Launch Plan:** 30‑day partnership activation  

            ---

            **Contact:**  
            Surprise Fakude, Director — Kaycore Creatives  
            Email: surprise@kaycorecreatives.com  
            Website: kaycorecreatives.com  
            """, unsafe_allow_html=True)

        with col_prop2:
            st.markdown("### Quick Stats")
            st.metric("Fit Score", f"{partner['kaycore_fit_score']:.1f}/10")
            st.metric("Est. Revenue", f"${partner['revenue_usd']:,.0f}")
            st.metric("Company Size", f"{partner['employees']} employees")
            st.metric("Priority", partner['partnership_priority'])

            st.markdown("---")
            st.markdown(f"**Website:**  \n{partner['website'] if pd.notna(partner['website']) else 'N/A'}")
            st.markdown(f"**Location:**  \n{partner['location_city']}, {partner['country']}")

            st.markdown("---")

            if st.button("Download PDF Proposal"):
                st.success("PDF generated! (Feature coming soon)")

            if st.button("Email Proposal"):
                st.success("Email sent! (Feature coming soon)")


# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
<strong>Kaycore Global Partnership Analytics Pro v1.0</strong><br>
Built by Suraj Raut & Saul Guzman | Data Science Interns 2026<br>
SA | UK | US | AU
</div>
""", unsafe_allow_html=True)
