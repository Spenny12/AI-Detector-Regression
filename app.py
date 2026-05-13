import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cloudscraper
from bs4 import BeautifulSoup
from transformers import pipeline
import io

# --- LOAD AI DETECTOR LOCALLY ---
@st.cache_resource(show_spinner="Downloading Oxidane AI Detector model...")
def load_ai_detector():
    """Downloads the model directly into memory."""
    # device=-1 forces it to run on CPU
    return pipeline("text-classification", model="Oxidane/tmr-ai-text-detector", device=-1)

# Initialize the model
detector_pipeline = load_ai_detector()

def scrape_text_from_url(url):
    """Scrapes paragraph text using Cloudscraper to bypass bot protection."""
    try:
        # Create a scraper that mimics a real Chrome browser
        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})

        response = scraper.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        # Security Check: Did we still hit a Captcha/Cloudflare wall?
        if "enable cookies" in text.lower() or "verify you are human" in text.lower() or "cloudflare" in text.lower():
             st.toast(f"⚠️ Bot protection blocked scraping for: {url}")
             return None

        # RoBERTa models process max 512 tokens. We truncate characters to prevent crashes.
        return text[:1500]
    except Exception as e:
        st.toast(f"Error scraping {url}: {e}")
        return None

def evaluate_ai_content_locally(text):
    """Evaluates text and returns the RAW probability (0.0 to 1.0)."""
    if not text or len(text.strip()) < 50:
        return None # Return None for empty text so it doesn't skew the math

    try:
        result = detector_pipeline(text)
        fake_score = 0.5

        if isinstance(result, list) and len(result) > 0:
            label_data = result[0]
            label_name = str(label_data.get('label', '')).lower()
            score = label_data.get('score', 0.5)

            # Identify if the highest confidence label means "AI"
            if label_name in ['ai', 'fake', 'label_1', '1', 'generated']:
                fake_score = score
            else:
                fake_score = 1.0 - score

        return fake_score # Return the raw float for curve grading

    except Exception as e:
        st.toast(f"Evaluation Error: {e}")
        return None

# --- EXCEL EXPORT FUNCTION ---
def generate_excel(df, r_squared, p_value, slope):
    """Creates a multi-sheet Excel file in memory."""
    output = io.BytesIO()

    summary_data = {
        "Metric": ["R-Squared", "P-Value", "Trend (Slope)"],
        "Value": [round(r_squared, 4), round(p_value, 4), round(slope, 2)],
        "Definition": [
            "How much of the click change is explained by the AI score (0 to 1).",
            "Statistical significance. < 0.05 is usually considered significant.",
            "Average change in clicks for every 1 point increase in AI score."
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Regression Summary', index=False)
        df.to_excel(writer, sheet_name='Scored Data', index=False)

    return output.getvalue()

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Content vs. Clicks Analyzer", layout="wide")

st.title("📈 AI Content vs. Organic Click Performance")
st.markdown("""
Upload a CSV containing your URLs and their Year-over-Year (YoY) click change.
This tool scans the pages, rates the presence of AI-generated content (**1-10**) using a curve-graded **Oxidane/tmr-ai-text-detector** score, and performs a linear regression.
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

            raw_scores = []
            total_urls = len(df)

            for index, row in df.iterrows():
                url = row['URL']
                status_text.text(f"Processing ({index + 1}/{total_urls}): {url}")

                # 1. Scrape
                scraped_text = scrape_text_from_url(url)

                # 2. Evaluate (Get raw float)
                score = evaluate_ai_content_locally(scraped_text)
                raw_scores.append(score)

                # Update progress
                progress_bar.progress((index + 1) / total_urls)

            status_text.text("✅ Evaluation complete! Grading on a curve...")

            # --- FIXED QUANTILE BINNING LOGIC ---
            df['Raw_AI_Prob'] = raw_scores

            # Fill failed scrapes with median site score
            median_score = df['Raw_AI_Prob'].median()
            df['Raw_AI_Prob'] = df['Raw_AI_Prob'].fillna(median_score)

            try:
                # Use pd.qcut to force the scores into 10 equal buckets
                bins = pd.qcut(df['Raw_AI_Prob'], q=10, duplicates='drop')

                # If all scores are identical, qcut drops all bins. We catch this to prevent a score of 0.
                if len(bins.cat.categories) == 0:
                    df['AI_Score'] = 5
                else:
                    df['AI_Score'] = bins.cat.codes + 1
            except Exception:
                # Fallback if there is zero variance in the data
                df['AI_Score'] = 5

            st.write("### Scored Data", df[['URL', 'Click_Change', 'Raw_AI_Prob', 'AI_Score']])

            # --- REGRESSION AND VISUALIZATION ---
            st.write("### Linear Regression Analysis")

            # Drop NaN values in Click_Change so Plotly doesn't crash
            plot_df = df.dropna(subset=['Click_Change'])

            if len(plot_df) > 1:
                # Prevent Plotly OLS crash if all X values are perfectly identical
                if plot_df['AI_Score'].nunique() > 1:
                    fig = px.scatter(
                        plot_df,
                        x="AI_Score",
                        y="Click_Change",
                        trendline="ols",
                        hover_data=["URL"],
                        title="Impact of AI Content on YoY Click Change",
                        labels={
                            "AI_Score": "AI Content Score (1 = Human, 10 = AI)",
                            "Click_Change": "Click Change YoY"
                        }
                    )
                else:
                    fig = px.scatter(
                        plot_df,
                        x="AI_Score",
                        y="Click_Change",
                        hover_data=["URL"],
                        title="Impact of AI Content on YoY Click Change",
                        labels={
                            "AI_Score": "AI Content Score (1 = Human, 10 = AI)",
                            "Click_Change": "Click Change YoY"
                        }
                    )
                    st.warning("All pages received the exact same AI Score, so a regression trendline cannot be drawn.")

                fig.update_xaxes(tickvals=list(range(1, 11)), range=[0.5, 10.5])
                fig.update_layout(template="plotly_white")
                fig.update_traces(marker=dict(size=10, opacity=0.7, color="#1f77b4"))

                st.plotly_chart(fig, use_container_width=True)

                # --- EXTRACT STATS & CREATE EXPORT ---
                # Only try to extract stats if the trendline actually generated
                if plot_df['AI_Score'].nunique() > 1:
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

                        # --- DOWNLOAD BUTTON ---
                        st.divider()
                        st.write("### Export Report")

                        excel_file = generate_excel(df, r_squared, p_value, slope)

                        st.download_button(
                            label="📥 Download Excel Data",
                            data=excel_file,
                            file_name="ai_impact_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
            else:
                st.error("Not enough valid 'Click_Change' data to perform a linear regression. Please ensure your CSV has numerical values in the Click_Change column.")
