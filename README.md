Data Insights Dashboard
A Streamlit-powered dashboard for automated data analysis and visualization. Users upload a CSV or Excel file, specify an analysis problem, and the app generates tailored visualizations (e.g., scatter plots, heatmaps, box plots) with actionable insights using Google Gemini AI. Designed for data scientists, it auto-selects columns, provides findings, and includes delete buttons for each visualization. Built with Python, Streamlit, Plotly, and Pandas.
Date: May 16, 2025
Features

File Upload: Supports CSV/Excel files (max 10MB).
Problem-Driven Analysis: Enter a problem statement (e.g., "Understand factors affecting exam scores") to guide visualizations.
Auto-Column Selection: No need to specify X/Y axes; the app selects columns based on data and problem.
Comprehensive Visualizations: Includes scatter, box, violin, bar, pie, heatmap, pairplot, and more.
AI Insights: Google Gemini provides actionable insights for each analysis.
Interactive UI: Delete unwanted visualizations with a single click.
Robust Error Handling: Handles file, API, and rendering errors gracefully.

Prerequisites

Python 3.8+
Google Gemini API key (sign up at Google Cloud)

Installation

Clone the Repository:
git clone <repository-url>
cd data-insights-dashboard


Install Dependencies:
pip install -r requirements.txt


Configure API Key:

Open app.py and replace "YOUR_API_KEY" with your Google Gemini API key:GEMINI_API_KEY = "your-actual-api-key"





Usage

Run the App:
streamlit run app.py


Open http://localhost:8501 in your browser.


Upload Data:

Upload a CSV/Excel file with numeric (e.g., exam_scores, study_hours) and categorical (e.g., parental_education) columns.
Example dataset:study_hours,attendance,exam_scores,social_media_hours,mental_health_rating,parental_education
5,90,85,2,6,High
3,70,60,4,4,Medium
6,95,90,1,7,High




Specify Problem:

Enter an analysis problem (e.g., "Understand factors affecting exam scores").
Optionally, filter by date range if a date column exists.


Explore Visualizations:

View auto-generated plots with findings (e.g., correlations between study_hours and exam_scores).
Delete unwanted visualizations using the "Delete Visualization" button.



Example Output

Problem: "Understand factors affecting exam scores"
Visualizations:
Scatter Plot: study_hours vs. exam_scores
Correlation Heatmap: Numeric columns
Box Plot: Distribution of exam_scores, mental_health_rating
Grouped Bar: exam_scores by parental_education


Insights: "Potential correlations between study hours, attendance, and exam scores…"

Troubleshooting

Duplicate ID Errors: Ensure st.plotly_chart calls use unique key parameters (fixed in code).
Gemini API Errors:
Verify API key validity.
Check logs (console or logfile) for raw response issues.
Test API independently:import google.generativeai as genai
genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-1.5-flash')
print(model.generate_content("Test").text)




Visualization Issues:
Confirm dataset has expected columns and types.
Clear Streamlit cache:rm -rf ~/.streamlit/cache





Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes (git commit -m "Add new feature").
Push to branch (git push origin feature/new-feature).
Open a pull request.


Contact
Built by Amresh

Built with Streamlit & Google Gemini | May 2025
