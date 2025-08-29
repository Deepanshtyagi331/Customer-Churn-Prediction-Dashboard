"""
Customer Churn Prediction Dashboard
==================================

Interactive Streamlit dashboard for customer churn prediction,
customer availability tracking, and product recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Streamlit version compatibility
try:
    # For Streamlit >= 1.18.0
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except AttributeError:
    # For Streamlit < 1.18.0
    cache_data = st.cache
    cache_resource = lambda **kwargs: st.cache(allow_output_mutation=True, **kwargs)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.churn_model import ChurnPredictionModel
    from models.recommendation_engine import ProductRecommendationEngine
    from data.data_generator import CustomerDataGenerator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.high-risk {
    color: #d62728;
    font-weight: bold;
}
.medium-risk {
    color: #ff7f0e;
    font-weight: bold;
}
.low-risk {
    color: #2ca02c;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@cache_data
@cache_data
def load_customer_data():
    """Load customer data with caching."""
    data_path = "data/customer_data.csv"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    try:
        df = pd.read_csv(data_path)
        return df, "loaded"
    except FileNotFoundError:
        try:
            from data.data_generator import CustomerDataGenerator
            generator = CustomerDataGenerator(n_customers=1000)
            df = generator.generate_complete_dataset()
            df.to_csv(data_path, index=False)
            return df, "generated"
        except Exception as e:
            # Return empty DataFrame with required columns for demo
            df = pd.DataFrame({
                'customer_id': ['DEMO_001'],
                'plan_type': ['Basic'],
                'churned': [0],
                'churn_probability': [0.5],
                'monthly_fee': [39.99],
                'satisfaction_score': [7.0],
                'tenure_months': [12],
                'monthly_usage_hours': [25.0],
                'support_tickets': [1],
                'last_activity_days': [5.0],
                'customer_value': [500.0],
                'age': [35],
                'gender': ['Male'],
                'city': ['New York'],
                'industry': ['Technology'],
                'total_spent': [960.0]
            })
            return df, f"error: {str(e)}"

@cache_resource()
def load_models():
    """Load ML models with caching."""
    try:
        from models.churn_model import ChurnPredictionModel
        from models.recommendation_engine import ProductRecommendationEngine
        
        churn_model = ChurnPredictionModel()
        recommendation_engine = ProductRecommendationEngine()
        
        # Try to load existing models
        churn_loaded = churn_model.load_model("models/churn_model.pkl")
        rec_loaded = recommendation_engine.load_model("models/recommendation_model.pkl")
        
        return churn_model, recommendation_engine, churn_loaded, rec_loaded
        
    except ImportError:
        # Return None objects if models can't be imported
        return None, None, False, False

def main():
    """Main dashboard application."""
    
    try:
        # Header
        st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Dashboard</h1>', 
                    unsafe_allow_html=True)
        
        # Load data and models
        with st.spinner("Loading data and models..."):
            try:
                df, data_status = load_customer_data()
                
                # Show data loading status
                if data_status == "generated":
                    st.warning("Customer data not found. Generated new dataset...")
                elif data_status.startswith("error:"):
                    st.error(f"Error loading data: {data_status[7:]}. Using demo data.")
                
                churn_model, recommendation_engine, churn_loaded, rec_loaded = load_models()
                
            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")
                # Create minimal demo data
                df = pd.DataFrame({
                    'customer_id': ['DEMO_001', 'DEMO_002'],
                    'plan_type': ['Basic', 'Premium'],
                    'churned': [0, 1],
                    'churn_probability': [0.3, 0.8],
                    'monthly_fee': [39.99, 79.99],
                    'satisfaction_score': [7.0, 4.0],
                    'tenure_months': [12, 8],
                    'monthly_usage_hours': [25.0, 10.0],
                    'support_tickets': [1, 5],
                    'last_activity_days': [5.0, 30.0],
                    'customer_value': [500.0, 640.0],
                    'age': [35, 42],
                    'gender': ['Male', 'Female'],
                    'city': ['New York', 'Los Angeles'],
                    'industry': ['Technology', 'Finance'],
                    'total_spent': [960.0, 1280.0]
                })
                churn_model, recommendation_engine, churn_loaded, rec_loaded = None, None, False, False
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["📊 Overview", "👤 Customer Analysis", "🔮 Churn Prediction", 
             "🛍️ Product Recommendations", "📈 Analytics", "⚙️ Model Management"]
        )
        
        # Main content based on selected page
        try:
            if page == "📊 Overview":
                show_overview(df)
            elif page == "👤 Customer Analysis":
                show_customer_analysis(df)
            elif page == "🔮 Churn Prediction":
                show_churn_prediction(df, churn_model, churn_loaded)
            elif page == "🛍️ Product Recommendations":
                show_product_recommendations(df, recommendation_engine, rec_loaded)
            elif page == "📈 Analytics":
                show_analytics(df)
            elif page == "⚙️ Model Management":
                show_model_management(churn_model, recommendation_engine)
        except Exception as e:
            st.error(f"Error displaying page '{page}': {str(e)}")
            st.info("Please try refreshing the page or selecting a different page.")
    
    except Exception as e:
        st.error(f"Critical error in application: {str(e)}")
        st.info("Please refresh the page. If the problem persists, check the system setup.")

def show_overview(df):
    """Display overview dashboard."""
    st.header("📊 Customer Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_rate = df['churned'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_revenue = df['monthly_fee'].mean()
        st.metric("Avg Monthly Revenue", f"${avg_revenue:.0f}")
    
    with col4:
        avg_satisfaction = df['satisfaction_score'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/10")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate by Plan Type")
        churn_by_plan = df.groupby('plan_type')['churned'].mean().reset_index()
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(churn_by_plan, x='plan_type', y='churned',
                        title="Churn Rate by Plan Type",
                        color='churned', color_continuous_scale='RdYlBu_r')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Matplotlib fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            churn_by_plan_series = df.groupby('plan_type')['churned'].mean()
            churn_by_plan_series.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Churn Rate by Plan Type")
            ax.set_ylabel("Churn Rate")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    
    with col2:
        st.subheader("Customer Distribution by Tenure")
        
        if PLOTLY_AVAILABLE:
            fig = px.histogram(df, x='tenure_months', nbins=20,
                              title="Customer Distribution by Tenure (Months)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Matplotlib fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df['tenure_months'], bins=20, color='lightblue', edgecolor='black')
            ax.set_title("Customer Distribution by Tenure (Months)")
            ax.set_xlabel("Tenure (Months)")
            ax.set_ylabel("Number of Customers")
            st.pyplot(fig)
    
    # Recent activity and alerts
    st.subheader("🚨 Customer Alerts")
    
    # Simulate real-time alerts
    high_risk_customers = df[df['churn_probability'] > 0.7].head(5)
    
    if len(high_risk_customers) > 0:
        st.warning(f"⚠️ {len(high_risk_customers)} customers at high risk of churning!")
        
        for _, customer in high_risk_customers.iterrows():
            with st.expander(f"Customer {customer['customer_id']} - Risk: {customer['churn_probability']:.1%}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Plan:** {customer['plan_type']}")
                    st.write(f"**Tenure:** {customer['tenure_months']} months")
                with col2:
                    st.write(f"**Usage:** {customer['monthly_usage_hours']:.1f} hours")
                    st.write(f"**Satisfaction:** {customer['satisfaction_score']:.1f}/10")
                with col3:
                    st.write(f"**Last Activity:** {customer['last_activity_days']:.0f} days ago")
                    st.write(f"**Support Tickets:** {customer['support_tickets']}")

def show_customer_analysis(df):
    """Display customer analysis page."""
    st.header("👤 Customer Analysis")
    
    # Customer selector
    customer_ids = df['customer_id'].tolist()
    selected_customer = st.selectbox("Select a customer:", customer_ids)
    
    if selected_customer:
        customer_data = df[df['customer_id'] == selected_customer].iloc[0]
        
        # Customer info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Customer Profile")
            st.write(f"**ID:** {customer_data['customer_id']}")
            st.write(f"**Age:** {customer_data.get('age', 'N/A')}")
            st.write(f"**Gender:** {customer_data.get('gender', 'N/A')}")
            st.write(f"**City:** {customer_data.get('city', 'N/A')}")
            st.write(f"**Industry:** {customer_data.get('industry', 'N/A')}")
        
        with col2:
            st.subheader("Account Information")
            st.write(f"**Plan:** {customer_data['plan_type']}")
            st.write(f"**Monthly Fee:** ${customer_data['monthly_fee']:.2f}")
            st.write(f"**Tenure:** {customer_data['tenure_months']} months")
            st.write(f"**Total Spent:** ${customer_data.get('total_spent', customer_data['monthly_fee'] * customer_data['tenure_months']):.2f}")
        
        with col3:
            st.subheader("Engagement Metrics")
            st.write(f"**Usage Hours:** {customer_data['monthly_usage_hours']:.1f}")
            st.write(f"**Satisfaction:** {customer_data['satisfaction_score']:.1f}/10")
            st.write(f"**Support Tickets:** {customer_data['support_tickets']}")
            
            # Risk level
            risk_prob = customer_data['churn_probability']
            if risk_prob > 0.7:
                risk_level = "🔴 High Risk"
                risk_class = "high-risk"
            elif risk_prob > 0.4:
                risk_level = "🟡 Medium Risk"
                risk_class = "medium-risk"
            else:
                risk_level = "🟢 Low Risk"
                risk_class = "low-risk"
            
            st.markdown(f'<p class="{risk_class}">Risk Level: {risk_level}</p>', 
                       unsafe_allow_html=True)
        
        # Customer behavior chart
        st.subheader("Customer Behavior Analysis")
        
        # Create radar chart for customer profile
        categories = ['Usage', 'Satisfaction', 'Tenure', 'Value', 'Engagement']
        values = [
            min(customer_data['monthly_usage_hours'] / 50 * 10, 10),
            customer_data['satisfaction_score'],
            min(customer_data['tenure_months'] / 36 * 10, 10),
            min(customer_data['customer_value'] / 200 * 10, 10),
            customer_data['feature_usage_score'] / 10
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Customer Profile'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="Customer Profile Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_churn_prediction(df, churn_model, model_loaded):
    """Display churn prediction page."""
    st.header("🔮 Churn Prediction")
    
    if not model_loaded:
        st.error("Churn prediction model not loaded. Please train the model first.")
        return
    
    # Prediction mode
    mode = st.radio("Prediction Mode:", ["Single Customer", "Batch Prediction"])
    
    if mode == "Single Customer":
        st.subheader("Single Customer Prediction")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 70, 35)
            income = st.number_input("Annual Income", 25000, 200000, 60000)
            tenure = st.slider("Tenure (months)", 0, 60, 12)
            monthly_fee = st.number_input("Monthly Fee", 20, 500, 80)
            plan_type = st.selectbox("Plan Type", ["Basic", "Premium", "Pro", "Enterprise"])
        
        with col2:
            usage_hours = st.slider("Monthly Usage Hours", 0, 100, 25)
            satisfaction = st.slider("Satisfaction Score", 0.0, 10.0, 7.0)
            support_tickets = st.slider("Support Tickets", 0, 10, 1)
            last_activity = st.slider("Days Since Last Activity", 0, 60, 5)
            feature_usage = st.slider("Feature Usage Score", 0, 100, 70)
        
        if st.button("Predict Churn"):
            # Create customer profile
            customer_profile = {
                'age': age,
                'income': income,
                'tenure_months': tenure,
                'monthly_fee': monthly_fee,
                'plan_type': plan_type,
                'monthly_usage_hours': usage_hours,
                'satisfaction_score': satisfaction,
                'support_tickets': support_tickets,
                'last_activity_days': last_activity,
                'feature_usage_score': feature_usage,
                'gender': 'Male',  # Default values
                'city': 'New York',
                'industry': 'Technology'
            }
            
            try:
                prediction = churn_model.predict_churn(customer_profile)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    churn_prob = prediction['churn_probability']
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                
                with col2:
                    risk_level = prediction['risk_level']
                    st.metric("Risk Level", risk_level)
                
                with col3:
                    confidence = prediction['confidence']
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = churn_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    else:  # Batch Prediction
        st.subheader("Batch Prediction Results")
        
        # Calculate churn probabilities for all customers
        high_risk = df[df['churn_probability'] > 0.7]
        medium_risk = df[(df['churn_probability'] > 0.4) & (df['churn_probability'] <= 0.7)]
        low_risk = df[df['churn_probability'] <= 0.4]
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Risk Customers", len(high_risk), 
                     delta=f"{len(high_risk)/len(df)*100:.1f}%")
        
        with col2:
            st.metric("Medium Risk Customers", len(medium_risk),
                     delta=f"{len(medium_risk)/len(df)*100:.1f}%")
        
        with col3:
            st.metric("Low Risk Customers", len(low_risk),
                     delta=f"{len(low_risk)/len(df)*100:.1f}%")
        
        # Risk distribution chart
        risk_data = pd.DataFrame({
            'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
            'Count': [len(high_risk), len(medium_risk), len(low_risk)],
            'Color': ['red', 'orange', 'green']
        })
        
        fig = px.pie(risk_data, values='Count', names='Risk Level',
                    title="Customer Risk Distribution",
                    color_discrete_sequence=['red', 'orange', 'green'])
        st.plotly_chart(fig, use_container_width=True)
        
        # High risk customers table
        st.subheader("High Risk Customers (Top 10)")
        high_risk_display = high_risk.head(10)[
            ['customer_id', 'plan_type', 'tenure_months', 'satisfaction_score', 
             'churn_probability', 'monthly_fee']
        ].round(3)
        st.dataframe(high_risk_display, use_container_width=True)

def show_product_recommendations(df, recommendation_engine, model_loaded):
    """Display product recommendations page."""
    st.header("🛍️ Product Recommendations")
    
    # Initialize recommendation engine
    if not recommendation_engine.product_catalog is not None:
        recommendation_engine.create_product_catalog()
    
    # Customer selector
    customer_ids = df['customer_id'].tolist()
    selected_customer = st.selectbox("Select customer for recommendations:", customer_ids)
    
    if selected_customer:
        customer_data = df[df['customer_id'] == selected_customer].iloc[0]
        
        # Customer summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Summary")
            st.write(f"**Current Plan:** {customer_data['plan_type']}")
            st.write(f"**Monthly Fee:** ${customer_data['monthly_fee']:.2f}")
            st.write(f"**Satisfaction:** {customer_data['satisfaction_score']:.1f}/10")
            st.write(f"**Churn Risk:** {customer_data['churn_probability']:.1%}")
        
        with col2:
            st.subheader("Usage Patterns")
            st.write(f"**Monthly Usage:** {customer_data['monthly_usage_hours']:.1f} hours")
            st.write(f"**Support Tickets:** {customer_data['support_tickets']}")
            st.write(f"**Tenure:** {customer_data['tenure_months']} months")
            st.write(f"**Customer Value:** ${customer_data['customer_value']:.0f}")
        
        # Get recommendations
        customer_profile = customer_data.to_dict()
        recommendations = recommendation_engine.get_recommendations(
            customer_profile, n_recommendations=5
        )
        
        # Display recommendations
        st.subheader("🎯 Recommended Products")
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['product_name']} - Score: {rec['score']:.0f}/100"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Category:** {rec['category']}")
                    st.write(f"**Price:** ${rec['price']}")
                    st.write(f"**Features:**")
                    for feature in rec['features']:
                        st.write(f"  • {feature}")
                
                with col2:
                    st.write(f"**Recommendation Score:** {rec['score']:.0f}/100")
                    st.write(f"**Why recommended:** {rec['reason']}")
        
        # Product catalog
        st.subheader("📋 Full Product Catalog")
        catalog = recommendation_engine.product_catalog
        st.dataframe(catalog[['product_name', 'category', 'price', 'target_segment']], 
                    use_container_width=True)

def show_analytics(df):
    """Display analytics page."""
    st.header("📈 Advanced Analytics")
    
    # Analytics tabs
    tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Trend Analysis", "Feature Importance"])
    
    with tab1:
        st.subheader("Customer Segmentation Analysis")
        
        # Segment customers based on value and risk
        df['segment'] = 'Unknown'
        df.loc[(df['customer_value'] > 150) & (df['churn_probability'] < 0.3), 'segment'] = 'Champions'
        df.loc[(df['customer_value'] > 100) & (df['churn_probability'] < 0.5), 'segment'] = 'Loyal Customers'
        df.loc[(df['customer_value'] < 50) & (df['churn_probability'] > 0.7), 'segment'] = 'At Risk'
        df.loc[(df['tenure_months'] < 6), 'segment'] = 'New Customers'
        df.loc[df['segment'] == 'Unknown', 'segment'] = 'Regular Customers'
        
        # Segment distribution
        segment_counts = df['segment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment characteristics
        segment_stats = df.groupby('segment').agg({
            'customer_value': 'mean',
            'churn_probability': 'mean',
            'satisfaction_score': 'mean',
            'monthly_fee': 'mean'
        }).round(2)
        
        st.subheader("Segment Characteristics")
        st.dataframe(segment_stats, use_container_width=True)
    
    with tab2:
        st.subheader("Trend Analysis")
        
        # Simulate time series data
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
        churn_trend = np.random.normal(0.15, 0.02, len(dates))
        churn_trend = np.cumsum(np.random.normal(0, 0.005, len(dates))) + 0.15
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Churn_Rate': churn_trend,
            'New_Customers': np.random.poisson(200, len(dates)),
            'Revenue': np.random.normal(50000, 5000, len(dates))
        })
        
        # Churn rate trend
        fig = px.line(trend_df, x='Date', y='Churn_Rate',
                     title="Churn Rate Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Rate', 'New Customers', 'Revenue', 'Combined View'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['Churn_Rate'], 
                                name='Churn Rate'), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['New_Customers'], 
                                name='New Customers'), row=1, col=2)
        fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['Revenue'], 
                                name='Revenue'), row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        # Simulate feature importance
        features = ['satisfaction_score', 'monthly_usage_hours', 'support_tickets', 
                   'tenure_months', 'monthly_fee', 'last_activity_days', 
                   'feature_usage_score', 'customer_value']
        importance = np.random.exponential(0.1, len(features))
        importance = importance / importance.sum()
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance for Churn Prediction")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

def show_model_management(churn_model, recommendation_engine):
    """Display model management page."""
    st.header("⚙️ Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Prediction Model")
        
        if st.button("Train Churn Model"):
            with st.spinner("Training churn prediction model..."):
                try:
                    # Load data and train model
                    X, y, df = churn_model.load_and_preprocess_data()
                    churn_model.build_models(X, y)
                    churn_model.save_model()
                    st.success("Churn model trained and saved successfully!")
                except Exception as e:
                    st.error(f"Error training model: {e}")
        
        if st.button("Load Churn Model"):
            if churn_model.load_model():
                st.success("Churn model loaded successfully!")
            else:
                st.error("Failed to load churn model.")
    
    with col2:
        st.subheader("Recommendation Engine")
        
        if st.button("Initialize Recommendation Engine"):
            with st.spinner("Initializing recommendation engine..."):
                try:
                    recommendation_engine.create_product_catalog()
                    recommendation_engine.save_model()
                    st.success("Recommendation engine initialized!")
                except Exception as e:
                    st.error(f"Error initializing engine: {e}")
        
        if st.button("Load Recommendation Engine"):
            if recommendation_engine.load_model():
                st.success("Recommendation engine loaded successfully!")
            else:
                st.error("Failed to load recommendation engine.")
    
    # Data management
    st.subheader("Data Management")
    
    if st.button("Generate New Dataset"):
        with st.spinner("Generating new customer dataset..."):
            try:
                generator = CustomerDataGenerator(n_customers=5000)
                df = generator.generate_complete_dataset()
                df.to_csv("data/customer_data.csv", index=False)
                st.success("New dataset generated and saved!")
            except Exception as e:
                st.error(f"Error generating dataset: {e}")

if __name__ == "__main__":
    main()
