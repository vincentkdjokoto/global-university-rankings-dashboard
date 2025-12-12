import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Global University Rankings Explorer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #3498DB;
        font-weight: 600;
    }
    .region-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-box {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">🎓 Global University Rankings Explorer</h1>', unsafe_allow_html=True)
st.markdown("""
Explore and compare university profiles across different regions. This dashboard analyzes simulated data based on THE/QS ranking metrics to answer:
**"How do the profiles of top universities in the US differ from those in Europe and Africa?"**
""")

# Generate simulated university data (in a real project, you would load a CSV)
@st.cache_data
def generate_university_data():
    np.random.seed(42)
    n_universities = 300
    
    # Define regions with different characteristics
    regions = ['United States', 'Europe', 'Africa', 'Asia', 'Australia']
    region_dist = [100, 120, 30, 40, 10]
    
    data = []
    university_id = 1
    
    for region_idx, region in enumerate(regions):
        n_region = region_dist[region_idx]
        
        for i in range(n_region):
            # Base scores vary by region
            if region == 'United States':
                research_base = np.random.normal(85, 10)
                teaching_base = np.random.normal(80, 8)
                citations_base = np.random.normal(90, 5)
                industry_income_base = np.random.normal(88, 6)
                international_base = np.random.normal(75, 10)
            elif region == 'Europe':
                research_base = np.random.normal(80, 12)
                teaching_base = np.random.normal(85, 7)
                citations_base = np.random.normal(85, 8)
                industry_income_base = np.random.normal(75, 10)
                international_base = np.random.normal(85, 8)
            elif region == 'Africa':
                research_base = np.random.normal(60, 15)
                teaching_base = np.random.normal(70, 12)
                citations_base = np.random.normal(55, 15)
                industry_income_base = np.random.normal(50, 15)
                international_base = np.random.normal(65, 12)
            elif region == 'Asia':
                research_base = np.random.normal(75, 12)
                teaching_base = np.random.normal(78, 10)
                citations_base = np.random.normal(82, 9)
                industry_income_base = np.random.normal(80, 8)
                international_base = np.random.normal(70, 12)
            else:  # Australia
                research_base = np.random.normal(78, 10)
                teaching_base = np.random.normal(82, 8)
                citations_base = np.random.normal(80, 9)
                industry_income_base = np.random.normal(76, 10)
                international_base = np.random.normal(85, 7)
            
            # Add some randomness and ensure scores are between 0-100
            research = max(0, min(100, research_base + np.random.normal(0, 5)))
            teaching = max(0, min(100, teaching_base + np.random.normal(0, 5)))
            citations = max(0, min(100, citations_base + np.random.normal(0, 5)))
            industry_income = max(0, min(100, industry_income_base + np.random.normal(0, 5)))
            international = max(0, min(100, international_base + np.random.normal(0, 5)))
            
            # Overall score (weighted average similar to THE rankings)
            overall = (teaching * 0.3 + research * 0.3 + citations * 0.3 + 
                      industry_income * 0.025 + international * 0.075)
            
            # Generate university name based on region
            if region == 'United States':
                name = f"University of {np.random.choice(['California', 'Michigan', 'Texas', 'Illinois', 'New York', 'Massachusetts'])}"
                if i < 20:
                    name = f"{np.random.choice(['Harvard', 'Stanford', 'MIT', 'Caltech', 'Princeton'])} University"
            elif region == 'Europe':
                name = f"{np.random.choice(['University of', 'Technical University of', 'University College'])} {np.random.choice(['Oxford', 'Cambridge', 'London', 'Paris', 'Berlin', 'Zurich', 'Copenhagen'])}"
            elif region == 'Africa':
                name = f"{np.random.choice(['University of', 'Cape Town', 'Witwatersrand', 'Nairobi', 'Cairo', 'Ghana'])}"
                if i < 5:
                    name = f"University of {np.random.choice(['Cape Town', 'Pretoria', 'Nairobi'])}"
            elif region == 'Asia':
                name = f"{np.random.choice(['National University of', 'University of', 'Tokyo Institute of'])} {np.random.choice(['Singapore', 'Tokyo', 'Beijing', 'Seoul', 'Hong Kong'])}"
            else:
                name = f"{np.random.choice(['University of', 'Australian National', 'University of Sydney', 'University of Melbourne'])}"
            
            # Select a country based on region
            if region == 'Europe':
                country = np.random.choice(['United Kingdom', 'Germany', 'France', 'Switzerland', 'Netherlands', 'Sweden'])
            elif region == 'Africa':
                country = np.random.choice(['South Africa', 'Egypt', 'Kenya', 'Nigeria', 'Ghana', 'Morocco'])
            elif region == 'Asia':
                country = np.random.choice(['Singapore', 'Japan', 'China', 'South Korea', 'Hong Kong SAR'])
            elif region == 'Australia':
                country = 'Australia'
            else:
                country = 'United States'
            
            data.append({
                'id': university_id,
                'name': name,
                'country': country,
                'region': region,
                'overall_score': round(overall, 1),
                'teaching_score': round(teaching, 1),
                'research_score': round(research, 1),
                'citations_score': round(citations, 1),
                'industry_income_score': round(industry_income, 1),
                'international_outlook_score': round(international, 1),
                'student_staff_ratio': np.random.normal(12, 4),
                'international_students': np.random.normal(25, 15)
            })
            
            university_id += 1
    
    df = pd.DataFrame(data)
    # Create rank based on overall score
    df['world_rank'] = df['overall_score'].rank(ascending=False).astype(int)
    return df

# Load data
df = generate_university_data()

# Sidebar filters
st.sidebar.markdown('<h3 class="sub-header">🔍 Filter Universities</h3>', unsafe_allow_html=True)

# Region selection (multi-select)
selected_regions = st.sidebar.multiselect(
    'Select Regions:',
    options=df['region'].unique(),
    default=['United States', 'Europe', 'Africa']
)

# Country selection (dynamic based on regions)
if selected_regions:
    available_countries = df[df['region'].isin(selected_regions)]['country'].unique()
else:
    available_countries = df['country'].unique()

selected_countries = st.sidebar.multiselect(
    'Select Countries:',
    options=available_countries,
    default=list(available_countries)[:3] if len(available_countries) > 0 else []
)

# Rank range filter
rank_range = st.sidebar.slider(
    'World Rank Range:',
    min_value=1,
    max_value=int(df['world_rank'].max()),
    value=(1, 100),
    step=1
)

# Score filters
teaching_range = st.sidebar.slider(
    'Teaching Score Range:',
    min_value=0.0,
    max_value=100.0,
    value=(30.0, 100.0),
    step=1.0
)

research_range = st.sidebar.slider(
    'Research Score Range:',
    min_value=0.0,
    max_value=100.0,
    value=(30.0, 100.0),
    step=1.0
)

# Apply filters
filtered_df = df.copy()

if selected_regions:
    filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]

if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

filtered_df = filtered_df[
    (filtered_df['world_rank'] >= rank_range[0]) & 
    (filtered_df['world_rank'] <= rank_range[1]) &
    (filtered_df['teaching_score'] >= teaching_range[0]) & 
    (filtered_df['teaching_score'] <= teaching_range[1]) &
    (filtered_df['research_score'] >= research_range[0]) & 
    (filtered_df['research_score'] <= research_range[1])
]

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Regional Comparison", "🏫 University Explorer", "📈 Teaching vs Research", "🔍 Key Insights"])

with tab1:
    st.markdown('<h3 class="sub-header">Regional Performance Profiles</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart comparing regions
        if not filtered_df.empty and len(filtered_df['region'].unique()) > 0:
            region_means = filtered_df.groupby('region')[['teaching_score', 'research_score', 
                                                         'citations_score', 'industry_income_score',
                                                         'international_outlook_score']].mean().reset_index()
            
            fig_radar = go.Figure()
            
            colors = px.colors.qualitative.Set2
            for idx, region in enumerate(region_means['region'].unique()):
                region_data = region_means[region_means['region'] == region].iloc[0]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[region_data['teaching_score'], region_data['research_score'],
                       region_data['citations_score'], region_data['industry_income_score'],
                       region_data['international_outlook_score'], region_data['teaching_score']],
                    theta=['Teaching', 'Research', 'Citations', 'Industry Income', 
                          'International Outlook', 'Teaching'],
                    name=region,
                    fill='toself',
                    line_color=colors[idx % len(colors)],
                    opacity=0.7
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title='Regional Comparison: Performance Metrics',
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Bar chart: Average scores by region
        if not filtered_df.empty:
            region_avg = filtered_df.groupby('region')['overall_score'].mean().sort_values(ascending=False)
            
            fig_bar = px.bar(
                x=region_avg.index,
                y=region_avg.values,
                title=f'Average Overall Score by Region (Top {len(filtered_df)} Universities)',
                labels={'x': 'Region', 'y': 'Average Overall Score'},
                color=region_avg.index,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig_bar.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Display key metrics in columns
    st.markdown('<h3 class="sub-header">Key Regional Metrics</h3>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            us_avg = filtered_df[filtered_df['region'] == 'United States']['overall_score'].mean() if 'United States' in filtered_df['region'].values else 0
            st.metric("🇺🇸 US Average Score", f"{us_avg:.1f}" if us_avg > 0 else "N/A")
        
        with metrics_col2:
            europe_avg = filtered_df[filtered_df['region'] == 'Europe']['overall_score'].mean() if 'Europe' in filtered_df['region'].values else 0
            st.metric("🇪🇺 Europe Average Score", f"{europe_avg:.1f}" if europe_avg > 0 else "N/A")
        
        with metrics_col3:
            africa_avg = filtered_df[filtered_df['region'] == 'Africa']['overall_score'].mean() if 'Africa' in filtered_df['region'].values else 0
            st.metric("🇿🇦 Africa Average Score", f"{africa_avg:.1f}" if africa_avg > 0 else "N/A")
        
        with metrics_col4:
            avg_research = filtered_df['research_score'].mean()
            st.metric("Overall Research Avg", f"{avg_research:.1f}")

with tab2:
    st.markdown('<h3 class="sub-header">University Details</h3>', unsafe_allow_html=True)
    
    # Sort options
    sort_by = st.selectbox(
        'Sort by:',
        options=['world_rank', 'overall_score', 'research_score', 'teaching_score', 'country'],
        index=0
    )
    
    sort_order = st.radio('Order:', ['Ascending', 'Descending'], horizontal=True, index=1)
    
    # Sort dataframe
    display_df = filtered_df.sort_values(
        by=sort_by, 
        ascending=(sort_order == 'Ascending')
    ).head(50)  # Limit to top 50 for performance
    
    # Display university cards
    for idx, row in display_df.iterrows():
        with st.expander(f"{row['world_rank']}. {row['name']} ({row['country']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Score", f"{row['overall_score']:.1f}")
                st.metric("Teaching", f"{row['teaching_score']:.1f}")
            
            with col2:
                st.metric("Research", f"{row['research_score']:.1f}")
                st.metric("Citations", f"{row['citations_score']:.1f}")
            
            with col3:
                st.metric("International", f"{row['international_outlook_score']:.1f}")
                st.metric("Industry Income", f"{row['industry_income_score']:.1f}")
    
    # Show raw data option
    if st.checkbox('Show raw data table'):
        st.dataframe(display_df[['world_rank', 'name', 'country', 'region', 'overall_score', 
                                'teaching_score', 'research_score']])

with tab3:
    st.markdown('<h3 class="sub-header">Teaching vs Research Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot: Teaching vs Research
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df,
                x='teaching_score',
                y='research_score',
                color='region',
                size='overall_score',
                hover_name='name',
                title='Teaching Score vs Research Score',
                labels={'teaching_score': 'Teaching Score', 'research_score': 'Research Score'},
                trendline='ols',
                height=500
            )
            
            # Add correlation line and annotation
            correlation = filtered_df['teaching_score'].corr(filtered_df['research_score'])
            fig_scatter.add_annotation(
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                text=f"Correlation: {correlation:.3f}",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Distribution of teaching-research difference
        if not filtered_df.empty:
            filtered_df['teaching_research_diff'] = filtered_df['teaching_score'] - filtered_df['research_score']
            
            fig_dist = px.histogram(
                filtered_df,
                x='teaching_research_diff',
                color='region',
                nbins=30,
                title='Distribution: Teaching Score Minus Research Score',
                labels={'teaching_research_diff': 'Teaching - Research Difference'},
                barmode='overlay',
                height=500
            )
            
            fig_dist.add_vline(x=0, line_dash="dash", line_color="gray", 
                              annotation_text="Balance", annotation_position="top")
            
            # Calculate and show average differences by region
            diff_by_region = filtered_df.groupby('region')['teaching_research_diff'].mean().sort_values()
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Display summary stats
            st.markdown("**Average Teaching-Research Difference by Region:**")
            for region, diff in diff_by_region.items():
                st.write(f"- **{region}**: {diff:+.1f} (Teaching is {'higher' if diff > 0 else 'lower'})")

with tab4:
    st.markdown('<h3 class="sub-header">Key Insights: US vs Europe vs Africa</h3>', unsafe_allow_html=True)
    
    # Generate insights based on filtered data
    insights_df = df[df['region'].isin(['United States', 'Europe', 'Africa'])]
    
    if not insights_df.empty:
        # Calculate key metrics for each region
        region_stats = {}
        
        for region in ['United States', 'Europe', 'Africa']:
            region_data = insights_df[insights_df['region'] == region]
            if len(region_data) > 0:
                region_stats[region] = {
                    'avg_overall': region_data['overall_score'].mean(),
                    'avg_teaching': region_data['teaching_score'].mean(),
                    'avg_research': region_data['research_score'].mean(),
                    'avg_citations': region_data['citations_score'].mean(),
                    'avg_industry': region_data['industry_income_score'].mean(),
                    'avg_international': region_data['international_outlook_score'].mean(),
                    'top_10_count': len(region_data[region_data['world_rank'] <= 10]),
                    'top_100_count': len(region_data[region_data['world_rank'] <= 100]),
                }
        
        # Display insights
        st.markdown("""
        <div class="metric-box">
        <h4>🏆 Key Findings from Simulated Data:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🇺🇸 United States Profile:**
            - Typically shows **strong research output** and **high industry income**
            - Balanced teaching-research relationship
            - Lower international outlook compared to European counterparts
            - Dominates top rankings with comprehensive excellence
            
            **🇪🇺 European Profile:**
            - Often **stronger in teaching** and **international outlook**
            - Slightly lower industry income metrics
            - More diverse performance across countries
            - Historic institutions with strong citations
            """)
        
        with col2:
            st.markdown("""
            **🇿🇦 African Profile:**
            - Shows **strong teaching focus** relative to research output
            - Lower scores across most metrics but with rapid growth potential
            - Higher student-staff ratios on average
            - Emerging research capabilities with targeted investments
            
            **📈 Regional Relationships:**
            - Teaching-Research correlation varies by region
            - US institutions show strongest industry partnerships
            - European universities lead in internationalization
            - All regions show positive teaching-research correlation
            """)
        
        # Data table with comparisons
        st.markdown("**📊 Quantitative Comparison (Averages):**")
        
        comparison_data = []
        for region, stats in region_stats.items():
            comparison_data.append({
                'Region': region,
                'Overall Score': f"{stats['avg_overall']:.1f}",
                'Teaching': f"{stats['avg_teaching']:.1f}",
                'Research': f"{stats['avg_research']:.1f}",
                'Citations': f"{stats['avg_citations']:.1f}",
                'Industry Income': f"{stats['avg_industry']:.1f}",
                'International': f"{stats['avg_international']:.1f}",
                'Top 100 Universities': stats['top_100_count']
            })
        
        st.table(pd.DataFrame(comparison_data))

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard:** This interactive tool explores simulated university ranking data based on THE/QS methodology. 
In a real implementation, you would replace the simulated data with actual rankings data from THE, QS, or ARWU datasets.

**Educational Context:** This analysis helps policymakers, researchers, and students understand regional strengths 
and differences in higher education systems worldwide.
""")

# Instructions for running locally
with st.sidebar.expander("🚀 How to Run This App"):
    st.markdown("""
    1. Save files as `app.py` and `requirements.txt`
    2. Install dependencies: `pip install -r requirements.txt`
    3. Run the app: `streamlit run app.py`
    4. Open browser to `http://localhost:8501`
    
    **For real data:** Replace the `generate_university_data()` function with code to load actual THE/QS data.
    """)
