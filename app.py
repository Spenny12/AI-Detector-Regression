import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time
import google.generativeai as genai

# --- AI EVALUATION LOGIC (GEMINI) ---
def scrape_text_from_url(url):
    """Scrapes paragraph text from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        # We limit to ~5000 characters to give Gemini plenty of context
        # while keeping requests lightweight.
        return text[:5000]
    except Exception as e:
        st.toast(f"Error scraping {url}: {e}")
        return None

def evaluate_ai_content(text, model):
    """Sends text to Gemini API and returns a 1-10 score or None on error."""
    if not text or len(text.strip()) < 50:
        return None  # Clearly indicate that there wasn't enough text to evaluate

    try:
        prompt = f"""
        Analyze the following text for signs of AI generation (e.g., repetitive sentence structures,
        lack of burstiness, common AI idioms, predictable vocabulary).

        Rate the likelihood of it being AI-generated on a strict integer scale of 1 to 10:
        1 = Entirely Human
        10 = Entirely AI

        Output ONLY the integer number. Do not include any other words, punctuation, or explanation.

        Text to analyze:
        {text}
        """

        # Using explicit timeout for robustness
        response = model.generate_content(
            prompt,
            request_options={'timeout': 30}
        )

        # Clean the response to ensure we only grab the integer
        score_text = response.text.strip()
        score = int(''.join(filter(str.isdigit, score_text)))

        # Clamp the score between 1 and 10 just in case
        return max(1, min(10, score))

    except Exception as e:
        # We'll handle the display of the error in the main loop to avoid toast clutter
        raise e


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Content vs. Clicks Analyzer", layout="wide")

st.title("📈 AI Content vs. Organic Click Performance")
st.markdown("""
Upload a CSV containing your URLs and their Year-over-Year (YoY) click change.
This tool scans the pages, rates the presence of AI-generated content (1-10) using **Google Gemini**, and performs a linear regression.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. API Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio (aistudio.google.com)")

st.sidebar.header("2. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Columns: URL, Click_Change)", type=["csv"])

# --- MAIN APP FLOW ---
if uploaded_file is not None:
    if not gemini_api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar to proceed.")
    else:
        df = pd.read_csv(uploaded_file)

        if 'URL' not in df.columns or 'Click_Change' not in df.columns:
            st.error("CSV must contain exactly 'URL' and 'Click_Change' columns.")
        else:
            st.write("### Data Preview", df.head())

            if st.button("Run AI Evaluation & Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Configure the Gemini API client once
                try:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
                except Exception as e:
                    st.error(f"Failed to initialize Gemini API: {e}")
                    st.stop()

                ai_scores = []
                total_urls = len(df)

                for index, row in df.iterrows():
                    url = row['URL']
                    status_text.text(f"Processing ({index + 1}/{total_urls}): {url}")

                    # 1. Scrape
                    scraped_text = scrape_text_from_url(url)

                    if not scraped_text:
                        st.warning(f"⚠️ Could not extract text from {url}. Skipping AI evaluation.")
                        ai_scores.append(None)
                    else:
                        # 2. Evaluate
                        try:
                            score = evaluate_ai_content(scraped_text, model)
                            if score is None:
                                st.warning(f"⚠️ Not enough text on {url} for a reliable score (min 50 chars).")
                            ai_scores.append(score)
                        except Exception as e:
                            st.error(f"❌ Error evaluating {url}: {e}")
                            ai_scores.append(None)

                        # Rate limit protection (as requested by user)
                        time.sleep(1)

                    # Update progress
                    progress_bar.progress((index + 1) / total_urls)

                status_text.text("✅ Evaluation complete!")
                df['AI_Score'] = ai_scores

                # Filter out rows where AI evaluation failed for the regression
                df_clean = df.dropna(subset=['AI_Score'])

                if df_clean.empty:
                    st.error("No valid AI scores were generated. Cannot perform analysis.")
                else:
                    st.write("### Scored Data", df)
                    if len(df_clean) < len(df):
                        st.info(f"Note: {len(df) - len(df_clean)} rows were excluded from analysis due to missing scores.")

                    # --- REGRESSION AND VISUALIZATION ---
                    st.write("### Linear Regression Analysis")

                    fig = px.scatter(
                        df_clean,
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

                    # Enhance marker visibility for individual data points
                    fig.update_layout(template="plotly_white")
                    fig.update_traces(
                        marker=dict(
                            size=12,
                            opacity=0.8,
                            color="#1f77b4",
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        selector=dict(mode='markers')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # --- EXTRACT STATS ---
                    results = px.get_trendline_results(fig)
                    if not results.empty:
                        model_results = results.iloc[0]["px_fit_results"]

                        col1, col2, col3 = st.columns(3)

                        r_squared = model_results.rsquared
                        p_value = model_results.pvalues[1]
                        slope = model_results.params[1]

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
