import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time

# --- AI EVALUATION LOGIC ---
def scrape_text_from_url(url):
    """Scrapes paragraph text from a given URL."""
    try:
        # Added headers to simulate a real browser and avoid some basic bot blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        # Hugging Face's roberta model has a token limit. 
        # We truncate to ~1500 characters to be safe and avoid API errors.
        return text[:1500] 
    except Exception as e:
        st.toast(f"Error scraping {url}: {e}")
        return None

def evaluate_ai_content(text, api_token):
    """Sends text to Hugging Face API and returns a 1-10 score."""
    if not text or len(text.strip()) < 50:
        return 5 # Neutral score if there's not enough text to analyze
        
    # --- UPDATED: New, modern ChatGPT detector model ---
    API_URL = "https://router.huggingface.co/hf-inference/models/Hello-SimpleAI/chatgpt-detector-roberta"
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        # Small sleep to respect free tier rate limits
        time.sleep(1.5)
        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        # If the status code is not 200 (OK), don't try to parse JSON
        if not response.ok:
            st.toast(f"API Failed ({response.status_code}): {response.text[:100]}")
            return 5 # Neutral fallback

        result = response.json()

        # Hugging Face models sometimes need to "wake up" if unused recently
        if isinstance(result, dict) and 'estimated_time' in result:
            st.toast(f"Model warming up, waiting {int(result['estimated_time'])} seconds...")
            time.sleep(result['estimated_time'])
            response = requests.post(API_URL, headers=headers, json={"inputs": text})
            if not response.ok:
                return 5
            result = response.json()

        fake_score = 0.5 # Default to neutral

        # Handle the standard Hugging Face classification output format
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
             for label_data in result[0]:
                # --- UPDATED: Look for the 'ChatGPT' label instead of 'Fake' ---
                if label_data.get('label') in ['Fake', 'ChatGPT']:
                    fake_score = label_data.get('score', 0.5)
                    break
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
             for label_data in result:
                if label_data.get('label') in ['Fake', 'ChatGPT']:
                    fake_score = label_data.get('score', 0.5)
                    break

        # Convert percentage (0.0 - 1.0) to a 1-10 integer scale
        final_score = int(round(fake_score * 10))
        return max(1, final_score)

    except Exception as e:
        st.toast(f"Evaluation Error: {e}")
        return 5 # Neutral fallback on failure


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Content vs. Clicks Analyzer", layout="wide")

st.title("📈 AI Content vs. Organic Click Performance")
st.markdown("""
Upload a CSV containing your URLs and their Year-over-Year (YoY) click change. 
This tool scans the pages, rates the presence of AI-generated content (1-10) using Hugging Face, and performs a linear regression.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. API Configuration")
hf_token = st.sidebar.text_input("Hugging Face API Token", type="password", help="Get a free token from huggingface.co/settings/tokens")

st.sidebar.header("2. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Columns: URL, Click_Change)", type=["csv"])

# --- MAIN APP FLOW ---
if uploaded_file is not None:
    if not hf_token:
        st.warning("⚠️ Please enter your Hugging Face API Token in the sidebar to proceed.")
    else:
        df = pd.read_csv(uploaded_file)
        
        if 'URL' not in df.columns or 'Click_Change' not in df.columns:
            st.error("CSV must contain exactly 'URL' and 'Click_Change' columns.")
        else:
            st.write("### Data Preview", df.head())
            
            if st.button("Run AI Evaluation & Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                ai_scores = []
                total_urls = len(df)
                
                for index, row in df.iterrows():
                    url = row['URL']
                    status_text.text(f"Processing ({index + 1}/{total_urls}): {url}")
                    
                    # 1. Scrape
                    scraped_text = scrape_text_from_url(url)
                    
                    # 2. Evaluate
                    score = evaluate_ai_content(scraped_text, hf_token)
                    ai_scores.append(score)
                    
                    # Update progress
                    progress_bar.progress((index + 1) / total_urls)
                    
                status_text.text("✅ Evaluation complete!")
                df['AI_Score'] = ai_scores
                
                st.write("### Scored Data", df)
                
                # --- REGRESSION AND VISUALIZATION ---
                st.write("### Linear Regression Analysis")
                
                fig = px.scatter(
                    df, 
                    x="AI_Score", 
                    y="Click_Change", 
                    trendline="ols",
                    hover_data=["URL"],
                    title="Impact of AI Content on YoY Click Change",
                    labels={
                        "AI_Score": "AI Content Score (1 = Human, 10 = AI)",
                        "Click_Change": "Click Change YoY (%)"
                    }
                )
                
                fig.update_layout(template="plotly_white")
                fig.update_traces(marker=dict(size=10, opacity=0.7, color="#1f77b4"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- EXTRACT STATS ---
                results = px.get_trendline_results(fig)
                if not results.empty:
                    model = results.iloc[0]["px_fit_results"]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    r_squared = model.rsquared
                    p_value = model.pvalues[1]
                    slope = model.params[1]
                    
                    col1.metric("R-Squared", f"{r_squared:.4f}")
                    col2.metric("P-Value", f"{p_value:.4f}")
                    col3.metric("Trend (Slope)", f"{slope:.2f}")
                    
                    st.markdown("#### Interpretation:")
                    if p_value < 0.05:
                        if slope < 0:
                            st.warning(f"**Statistically significant negative relationship.** As the AI score increases, click performance decreases by {abs(slope):.2f} units per point.")
                        else:
                            st.success(f"**Statistically significant positive relationship.** As the AI score increases, click performance increases by {slope:.2f} units per point.")
                    else:
                        st.info("**No statistically significant relationship** (p >= 0.05). The AI score does not strongly correlate with the click change in this dataset.")
