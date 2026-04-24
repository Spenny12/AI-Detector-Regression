import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# --- LOAD AI DETECTOR LOCALLY ---
@st.cache_resource(show_spinner="Downloading Oxidane AI Detector model (this takes a minute on first boot)...")
def load_ai_detector():
    """
    Downloads the model directly into Streamlit's memory.
    @st.cache_resource ensures it only downloads once and stays in RAM.
    """
    # device=-1 forces it to run on CPU, which is required for Streamlit Cloud
    return pipeline("text-classification", model="Oxidane/tmr-ai-text-detector", device=-1)

# Initialize the model
detector_pipeline = load_ai_detector()

def scrape_text_from_url(url):
    """Scrapes paragraph text from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        # RoBERTa models process max 512 tokens. We truncate characters to prevent crashes.
        return text[:1500]
    except Exception as e:
        st.toast(f"Error scraping {url}: {e}")
        return None

def evaluate_ai_content_locally(text):
    """Evaluates text using the locally loaded Hugging Face model."""
    if not text or len(text.strip()) < 50:
        return 5 # Neutral score if there's not enough text

    try:
        # Run the text through the local model
        result = detector_pipeline(text)

        fake_score = 0.5 # Default neutral

        # Parse the output. The pipeline usually returns: [{'label': 'LABEL_1', 'score': 0.85}]
        if isinstance(result, list) and len(result) > 0:
            label_data = result[0]
            label_name = str(label_data.get('label', '')).lower()
            score = label_data.get('score', 0.5)

            # Identify if the highest confidence label means "AI" (usually LABEL_1 or 'fake')
            if label_name in ['ai', 'fake', 'label_1', '1', 'generated']:
                fake_score = score
            else:
                # If the highest confidence is "Human" (LABEL_0), invert the score for our 1-10 scale
                fake_score = 1.0 - score

        # Convert percentage (0.0 - 1.0) to a 1-10 integer scale
        final_score = int(round(fake_score * 10))
        return max(1, min(10, final_score)) # Clamp between 1 and 10

    except Exception as e:
        st.toast(f"Evaluation Error: {e}")
        return 5

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Content vs. Clicks Analyzer", layout="wide")

st.title("📈 AI Content vs. Organic Click Performance")
st.markdown("""
Upload a CSV containing your URLs and their Year-over-Year (YoY) click change.
This tool scans the pages, rates the presence of AI-generated content (1-10) using **Oxidane/tmr-ai-text-detector** (running locally!), and performs a linear regression.
""")

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Columns: URL, Click_Change)", type=["csv"])

# --- MAIN APP FLOW ---
if uploaded_file is not None:
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

                # 2. Evaluate (Instant, no API calls)
                score = evaluate_ai_content_locally(scraped_text)
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
