
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import google.generativeai as genai
import io
import logging
import json
from datetime import datetime
import sys
import os
import re

# 🔹 Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔹 Gemini API Configuration
GEMINI_API_KEY = "Your API key"  # Replace with actual key
#genai.configure(api_key="AIzaSyDwLiS2uHId79Lhn2mwdr7dhNHZXYoHZl0")  # Replace with your valid API key
#GEMINI_MODEL = "gemini-1.5-flash-001-tuning" 
try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-1.5-flash-001-tuning')
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}. Please check your API key.")
    logger.error(f"Gemini API config error: {e}")
    llm = None




# 🔹 Streamlit App Setup
try:
    st.set_page_config(page_title="Data Insights Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("📈 Data Insights Dashboard")
    st.markdown("""
    Upload a CSV or Excel file and specify your analysis problem. The dashboard will automatically generate visualizations to help data scientists understand the data.  
    **Date & Time:** May 16, 2025, 12:20 AM IST
    """)
except Exception as e:
    st.error(f"Failed to initialize Streamlit app: {e}")
    logger.error(f"Streamlit init error: {e}")
    sys.exit(1)

# 🔹 Initialize Session State
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'viz_counter' not in st.session_state:
    st.session_state.viz_counter = 0

# 🔹 Utility Functions
@st.cache_data
def load_data(file):
    """Load CSV or Excel file into a pandas DataFrame."""
    if not file:
        st.warning("No file uploaded.")
        return None
    try:
        # Validate file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            raise ValueError("File size exceeds 10MB limit.")
        
        # Read file based on extension
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(file, encoding='utf-8', encoding_errors='ignore')
        elif file_ext == '.xlsx':
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
        
        # Validate DataFrame
        if df.empty:
            raise ValueError("Uploaded file is empty.")
        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names detected.")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logger.error(f"File load error: {e}")
        return None

def generate_insights(df, problem):
    """Generate AI insights using Gemini model based on dataset and problem."""
    if not llm:
        return "Gemini API not available. Please check API configuration."
    try:
        if df is None or df.empty:
            raise ValueError("No valid data to analyze.")
        
        # Create summary
        summary = (
            f"Dataset Info:\n- Columns: {', '.join(df.columns)}\n- Rows: {len(df)}\n"
            f"Sample (3 rows):\n{df.head(3).to_string()}\n"
            f"Statistics:\n{df.describe().to_string()}\n"
            f"User Problem: {problem}"
        )
        
        # Prepare prompt
        prompt = (
            "As a data analysis expert, analyze this dataset summary and the user's problem. "
            "Provide concise insights (100-150 words) on trends, patterns, or anomalies relevant to the problem. "
            "Focus on actionable insights for data scientists.\n\n"
            f"{summary}\n\n"
            "Format response as JSON: ```json\n{\"insights\": \"string\"}\n```"
        )
        
        # Generate insights
        response = llm.generate_content(prompt)
        if not hasattr(response, 'text') or not response.text:
            logger.error("Empty or invalid Gemini API response")
            return "Failed to generate insights: Empty response from Gemini API."
        
        # Clean response (remove ```json markers and extra whitespace)
        cleaned_response = re.sub(r'^```json\n|\n```$', '', response.text).strip()
        logger.info(f"Cleaned Gemini response: {cleaned_response}")
        
        # Parse JSON response
        try:
            insights = json.loads(cleaned_response)["insights"]
            if not isinstance(insights, str) or not insights:
                raise ValueError("Invalid insights format.")
            return insights
        except json.JSONDecodeError as je:
            logger.error(f"JSON parse error: {je}, Cleaned Response: {cleaned_response}")
            return "Failed to generate insights: Invalid JSON response from Gemini. Please check API key or try again."
        except KeyError:
            logger.error(f"Missing 'insights' key in response: {cleaned_response}")
            return "Failed to generate insights: Invalid response format from Gemini."
        
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        return f"Failed to generate insights: {e}"

def parse_problem(problem, df):
    """Parse problem statement and data to determine visualization types and columns."""
    problem = problem.lower()
    viz_types = []
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if 'date' in df.columns and df['date'].dtype == 'object':
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if df['date'].notna().any():
                datetime_cols = ['date']
        except:
            pass
    
    # Keyword-based visualization selection (tailored to student performance dataset)
    if 'trend' in problem or 'over time' in problem:
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            viz_types.append(('time_series', datetime_cols[0], numeric_cols))
            if len(categorical_cols) > 0:
                viz_types.append(('area', datetime_cols[0], numeric_cols[0], categorical_cols[0]))
    if 'compare' in problem or 'by category' in problem or 'across' in problem or 'performance' in problem:
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            viz_types.append(('grouped_bar', categorical_cols[0], ['exam_scores', 'mental_health_rating']))
            if len(datetime_cols) > 0:
                viz_types.append(('stacked_bar', datetime_cols[0], 'exam_scores', categorical_cols[0]))
            viz_types.append(('pie', categorical_cols[0], 'exam_scores'))
    if 'distribution' in problem or 'spread' in problem or 'outlier' in problem:
        if len(numeric_cols) > 0:
            viz_types.append(('box', None, ['exam_scores', 'mental_health_rating', 'study_hours']))
            viz_types.append(('histogram', 'exam_scores', None))
            if len(categorical_cols) > 0:
                viz_types.append(('violin', categorical_cols[0], 'exam_scores'))
    if 'relationship' in problem or 'correlation' in problem or 'factor' in problem:
        if len(numeric_cols) >= 2:
            viz_types.append(('scatter', 'study_hours', 'exam_scores'))
            viz_types.append(('scatter', 'mental_health_rating', 'exam_scores'))
            viz_types.append(('heatmap', None, numeric_cols))
            viz_types.append(('pairplot', None, numeric_cols))
    
    # Default visualizations for data exploration
    if not viz_types:
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            viz_types.append(('grouped_bar', categorical_cols[0], ['exam_scores', 'mental_health_rating']))
            viz_types.append(('pie', categorical_cols[0], 'exam_scores'))
        if len(numeric_cols) >= 2:
            viz_types.append(('scatter', 'study_hours', 'exam_scores'))
            viz_types.append(('heatmap', None, numeric_cols))
        if len(numeric_cols) > 0:
            viz_types.append(('box', None, ['exam_scores', 'mental_health_rating', 'study_hours']))
    
    # Remove duplicates manually
    unique_viz_types = []
    seen = set()
    for vt in viz_types:
        vt_tuple = (vt[0], vt[1], tuple(vt[2]) if isinstance(vt[2], list) else vt[2], vt[3] if len(vt) > 3 else None)
        if vt_tuple not in seen:
            seen.add(vt_tuple)
            unique_viz_types.append(vt)
    
    return unique_viz_types

def generate_findings(df, viz_type, params):
    """Generate findings for a specific visualization."""
    try:
        findings = f"**Findings for {viz_type.replace('_', ' ').title()}**: "
        if viz_type == 'time_series':
            findings += f"Plots {', '.join(params['y'])} over {params['x']}. Identifies trends, seasonality, or anomalies in student performance over time."
        elif viz_type == 'grouped_bar':
            findings += f"Compares average {', '.join(params['y'])} across {params['x']}. Highlights how categories (e.g., parental education) affect exam scores or mental health."
        elif viz_type == 'stacked_bar':
            findings += f"Shows {params['y']} by {params['color']} over {params['x']}. Reveals category contributions to exam scores over time."
        elif viz_type == 'pie':
            findings += f"Illustrates {params['values']} share by {params['names']}. Larger slices indicate higher contributions to total exam scores."
        elif viz_type == 'scatter':
            findings += f"Plots {params['x']} vs {params['y']}. Clusters or patterns suggest correlations (e.g., study hours impacting exam scores)."
        elif viz_type == 'box':
            findings += f"Shows distribution of {', '.join(params['y'])}. Identifies outliers and spread in exam scores or mental health ratings."
        elif viz_type == 'histogram':
            findings += f"Displays distribution of {params['x']}. Reveals skewness, modality, or outliers in exam scores."
        elif viz_type == 'heatmap':
            findings += f"Shows correlations between {', '.join(params['columns'])}. Values near 1 or -1 indicate strong relationships (e.g., study hours and exam scores)."
        elif viz_type == 'violin':
            findings += f"Compares distribution of {params['y']} across {params['x'] or 'data'}. Shows density and spread of exam scores by category."
        elif viz_type == 'pairplot':
            findings += f"Plots pairwise relationships for {', '.join(params['columns'])}. Useful for exploring correlations and distributions among study habits and scores."
        elif viz_type == 'area':
            findings += f"Shows cumulative {params['y']} by {params['color'] or 'data'} over {params['x']}. Highlights growth patterns in exam scores by category."
        return findings
    except Exception as e:
        logger.warning(f"Findings generation error for {viz_type}: {e}")
        return f"**Findings for {viz_type.replace('_', ' ').title()}**: Unable to generate findings due to data issues."

def create_visualizations(df, viz_types):
    """Generate visualizations based on requested types and parameters."""
    visualizations = []
    try:
        if df is None or df.empty:
            raise ValueError("No valid data for visualizations.")
        
        for vt in viz_types:
            viz_type = vt[0]
            try:
                params = {
                    'x': vt[1] if len(vt) > 1 else None,
                    'y': vt[2] if len(vt) > 2 else None,
                    'color': vt[3] if len(vt) > 3 else None,
                    'names': vt[1] if viz_type == 'pie' else None,
                    'values': vt[2] if viz_type == 'pie' else None,
                    'columns': vt[2] if viz_type in ['heatmap', 'pairplot'] else None
                }
                
                if viz_type == 'time_series' and params['x'] and params['y']:
                    if df[params['x']].notna().any():
                        fig = px.line(df, x=params['x'], y=params['y'],
                                      title=f"Trends of {', '.join(params['y'])} Over Time")
                        findings = generate_findings(df, viz_type, params)
                        visualizations.append((f"Time Series Plot", fig, findings))
                
                elif viz_type == 'grouped_bar' and params['x'] and params['y']:
                    fig = px.bar(df.groupby(params['x'])[params['y']].mean().reset_index(),
                                 x=params['x'], y=params['y'], barmode='group',
                                 title=f"Average {', '.join(params['y'])} by {params['x']}")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Grouped Bar Chart", fig, findings))
                
                elif viz_type == 'stacked_bar' and params['x'] and params['y'] and params['color']:
                    fig = px.bar(df.groupby([params['x'], params['color']])[params['y']].sum().reset_index(),
                                 x=params['x'], y=params['y'], color=params['color'],
                                 title=f"{params['y']} by {params['color']} Over {params['x']}")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Stacked Bar Chart", fig, findings))
                
                elif viz_type == 'pie' and params['names'] and params['values']:
                    if len(df[params['names']].unique()) <= 20:
                        fig = px.pie(df.groupby(params['names'])[params['values']].sum().reset_index(),
                                     values=params['values'], names=params['names'],
                                     title=f"{params['values']} Share by {params['names']}")
                        findings = generate_findings(df, viz_type, params)
                        visualizations.append((f"Pie Chart", fig, findings))
                
                elif viz_type == 'scatter' and params['x'] and params['y']:
                    fig = px.scatter(df, x=params['x'], y=params['y'],
                                     title=f"{params['x']} vs {params['y']}")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Scatter Plot", fig, findings))
                
                elif viz_type == 'box' and params['y']:
                    fig = px.box(df, y=params['y'], title=f"Distribution of {', '.join(params['y'])}")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Box Plot", fig, findings))
                
                elif viz_type == 'histogram' and params['x']:
                    fig = px.histogram(df, x=params['x'], nbins=30,
                                       title=f"Distribution of {params['x']}")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Histogram", fig, findings))
                
                elif viz_type == 'heatmap' and params['columns']:
                    corr = df[params['columns']].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Correlation Heatmap", fig, findings))
                
                elif viz_type == 'violin' and params['y']:
                    if params['x']:
                        fig = px.violin(df, x=params['x'], y=params['y'], box=True,
                                        title=f"Distribution of {params['y']} by {params['x']}")
                    else:
                        fig = px.violin(df, y=params['y'], box=True,
                                        title=f"Distribution of {params['y']}")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Violin Plot", fig, findings))
                
                elif viz_type == 'pairplot' and params['columns']:
                    fig, ax = plt.subplots()
                    sns.pairplot(df[params['columns']], diag_kind='kde')
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Pairplot", buf, findings))
                    plt.close(fig)
                
                elif viz_type == 'area' and params['x'] and params['y']:
                    if params['color']:
                        fig = px.area(df.groupby([params['x'], params['color']])[params['y']].sum().reset_index(),
                                      x=params['x'], y=params['y'], color=params['color'],
                                      title=f"Cumulative {params['y']} by {params['color']}")
                    else:
                        fig = px.area(df.groupby(params['x'])[params['y']].sum().reset_index(),
                                      x=params['x'], y=params['y'],
                                      title=f"Cumulative {params['y']} Over Time")
                    findings = generate_findings(df, viz_type, params)
                    visualizations.append((f"Area Chart", fig, findings))
                
            except Exception as e:
                logger.warning(f"Error generating {viz_type}: {e}")
        
        return visualizations
    
    except Exception as e:
        logger.error(f"Visualization generation error: {e}")
        return []

def delete_visualization(viz_id):
    """Remove a visualization from session state."""
    try:
        st.session_state.visualizations = [viz for viz in st.session_state.visualizations if viz[0] != viz_id]
    except Exception as e:
        st.error(f"Error deleting visualization: {e}")
        logger.error(f"Delete visualization error: {e}")

# 🔹 Sidebar: File Upload and Options
try:
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        #st.markdown("""
       # **Instructions:**  
        #- Upload a clean file with numeric/categorical columns.  
        #- Ensure clear headers for accurate analysis.  
        #- Max file size: 10MB.  
        #- After uploading, specify your analysis problem.
        #""")
        
        # Load data and show options
        df = None
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.header("Analysis Options")
                problem = st.text_input("Enter your analysis problem (e.g., 'Understand factors affecting exam scores')", "")
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        if df['date'].notna().any():
                            date_range = st.date_input("Select Date Range",
                                                      [df['date'].min(), df['date'].max()])
                    except Exception as e:
                        st.sidebar.warning(f"Date filter error: {e}")
        
        st.markdown("""
        **Instructions:**  
        - Upload a clean file with numeric/categorical columns.  
        - Ensure clear headers for accurate analysis.  
        - Max file size: 10MB.  
        - After uploading, specify your analysis problem.
        """)
except Exception as e:
    st.error(f"Sidebar setup error: {e}")
    logger.error(f"Sidebar error: {e}")

# 🔹 Main Content
try:
    if uploaded_file:
        # Load and display data
        if df is not None:
            # Apply date filter
            if 'date' in df.columns and 'date_range' in locals() and date_range:
                try:
                    df = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                            (df['date'] <= pd.to_datetime(date_range[1]))]
                    if df.empty:
                        st.warning("No data in selected date range.")
                except Exception as e:
                    st.warning(f"Date filter error: {e}")
                    logger.warning(f"Date filter error: {e}")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Problem input and visualization generation
            if 'problem' in locals() and problem:
                st.subheader("Analysis Problem")
                st.write(f"**Problem:** {problem}")
                
                # Generate insights
                with st.spinner("Generating AI insights..."):
                    insights = generate_insights(df, problem)
                st.subheader("AI Insights")
                st.markdown(f"**Gemini Analysis:** {insights}")
                
                # Generate visualizations based on problem
                with st.spinner("Generating visualizations..."):
                    viz_types = parse_problem(problem, df)
                    new_visualizations = create_visualizations(df, viz_types)
                    
                    # Assign unique IDs to new visualizations
                    for title, viz, findings in new_visualizations:
                        viz_id = f"viz_{st.session_state.viz_counter}"
                        st.session_state.visualizations.append((viz_id, title, viz, findings))
                        st.session_state.viz_counter += 1
                
                # Display visualizations with delete buttons
                st.subheader("Visualizations")
                if st.session_state.visualizations:
                    for viz_id, title, viz, findings in st.session_state.visualizations:
                        try:
                            with st.container():
                                st.markdown(f"### {title}")
                                if isinstance(viz, io.BytesIO):
                                    st.image(viz, use_column_width=True)
                                else:
                                    st.plotly_chart(viz, use_container_width=True, key=f"plotly_{viz_id}")
                                st.markdown(findings)
                                if st.button("Delete Visualization", key=f"delete_{viz_id}"):
                                    delete_visualization(viz_id)
                                    st.rerun()  # Refresh to update UI
                        except Exception as e:
                            st.warning(f"Error displaying {title}: {e}")
                            logger.warning(f"Display error for {title}: {e}")
                else:
                    st.warning("No visualizations generated. Check problem statement or data.")
            else:
                st.info("Please enter an analysis problem to generate visualizations.")
        else:
            st.error("Failed to load data. Please check the file and try again.")
    
    else:
        st.info("Please upload a CSV or Excel file to begin analysis.")

# 🔹 Footer
    st.markdown("---")
    st.markdown("Built with Streamlit & Google Gemini | Amresh | May 2025")
except Exception as e:
    st.error(f"Main content error: {e}")
    logger.error(f"Main content error: {e}")
