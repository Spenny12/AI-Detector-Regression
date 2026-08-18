import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from curl_cffi import requests as curl_requests
from bs4 import BeautifulSoup
from transformers import pipeline
import io
import re

# --- LOAD AI DETECTOR LOCALLY ---
@st.cache_resource(show_spinner="Downloading Oxidane AI Detector model...")
def load_ai_detector():
    """Downloads the model directly into memory."""
    # device=-1 forces it to run on CPU
    return pipeline("text-classification", model="Oxidane/tmr-ai-text-detector", device=-1)

# Initialize the model
detector_pipeline = load_ai_detector()

def scrape_text_from_url(url):
    """Scrapes paragraph text using curl_cffi to bypass strict TLS/SSL protection."""
    try:
        response = curl_requests.get(url, impersonate="chrome", timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        if "enable cookies" in text.lower() or "verify you are human" in text.lower() or "cloudflare" in text.lower():
             st.toast(f"⚠️ Bot protection blocked scraping for: {url}")
             return None

        # Return the complete text so the deep scan can evaluate the entire page
        return text
    except Exception as e:
        st.toast(f"Error scraping {url}: {e}")
        return None

        # Truncate to prevent token crashes on the main evaluation
        return text[:1500]
    except Exception as e:
        st.toast(f"Error scraping {url}: {e}")
        return None

def evaluate_ai_content_locally(text):
    """Evaluates a representative sample of the text and returns the RAW probability."""
    if not text or len(text.strip()) < 50:
        return None

    try:
        # If the text is short enough, just evaluate the whole thing
        if len(text) <= 2500:
            eval_text = text
        else:
            # Grab a sample from the top, middle, and bottom of the page
            head = text[:800]

            mid_start = len(text) // 2 - 400
            mid_end = len(text) // 2 + 400
            middle = text[mid_start:mid_end]

            tail = text[-800:]

            eval_text = f"{head} {middle} {tail}"

        # Force the tokenizer to truncate if we accidentally exceed 512 tokens
        result = detector_pipeline(eval_text, truncation=True, max_length=512)
        fake_score = 0.5

        if isinstance(result, list) and len(result) > 0:
            label_data = result[0]
            label_name = str(label_data.get('label', '')).lower()
            score = label_data.get('score', 0.5)

            if label_name in ['ai', 'fake', 'label_1', '1', 'generated']:
                fake_score = score
            else:
                fake_score = 1.0 - score

        return fake_score

    except Exception as e:
        st.toast(f"Evaluation Error: {e}")
        return None

def get_highly_likely_ai_sentences(text, threshold=0.80):
    """Splits text, evaluates, and merges contiguous AI chunks into readable blocks."""
    if not text:
        return []

    # 1. Split and group into evaluation chunks
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not raw_sentences:
        return []

    chunks = []
    current_chunk = ""

    for sentence in raw_sentences:
        current_chunk = (current_chunk + " " + sentence).strip()
        if len(current_chunk) >= 200:
            chunks.append(current_chunk)
            current_chunk = ""

    if current_chunk:
        if len(current_chunk) >= 100 or not chunks:
            chunks.append(current_chunk)
        else:
            chunks[-1] += " " + current_chunk

    try:
        # 2. Evaluate all chunks with truncation enabled to prevent token crashes
        results = detector_pipeline(chunks, truncation=True, max_length=512)

        # 3. --- NEW LOGIC: Merge Contiguous AI Blocks ---
        merged_ai_blocks = []
        active_block_text = []
        active_block_scores = []

        for chunk, res in zip(chunks, results):
            label_name = str(res.get('label', '')).lower()
            score = res.get('score', 0.5)

            fake_score = score if label_name in ['ai', 'fake', 'label_1', '1', 'generated'] else 1.0 - score

            if fake_score >= threshold:
                # Chunk is AI: Add it to the running block
                active_block_text.append(chunk)
                active_block_scores.append(fake_score)
            else:
                # Chunk is Human: Break the chain and save the block if one exists
                if active_block_text:
                    avg_score = sum(active_block_scores) / len(active_block_scores)
                    merged_ai_blocks.append({
                        "score": avg_score,
                        "text": " ".join(active_block_text) # Stitch back into a normal paragraph
                    })
                    # Reset the running block
                    active_block_text = []
                    active_block_scores = []

        # Catch any lingering AI block at the very end of the page
        if active_block_text:
            avg_score = sum(active_block_scores) / len(active_block_scores)
            merged_ai_blocks.append({
                "score": avg_score,
                "text": " ".join(active_block_text)
            })

        return merged_ai_blocks

    except Exception as e:
        st.toast(f"Chunk Extraction Error: {e}")
        return []

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
This tool scans the pages, rates the overall presence of AI-generated content (**1-10**), performs a linear regression, and **extracts specific sentences flagged as AI** for pages that exceed your defined threshold.
""")

st.sidebar.header("Data Input & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Columns: URL, Click_Change)", type=["csv"])

st.sidebar.divider()
st.sidebar.markdown("### Deep Scan Settings")
st.sidebar.markdown("Only extract specific AI sentences if the overall page AI probability is above this threshold.")
deep_scan_threshold_ui = st.sidebar.slider("Page AI Threshold for Deep Scan (%)", min_value=0, max_value=100, value=70, step=5)
deep_scan_threshold = deep_scan_threshold_ui / 100.0

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
            extracted_ai_texts_ui = [] # Simple strings for the Streamlit UI
            master_flagged_list = []   # Detailed data for the Excel export
            total_urls = len(df)

            for index, row in df.iterrows():
                url = row['URL']
                status_text.text(f"Processing ({index + 1}/{total_urls}): {url}")

                scraped_text = scrape_text_from_url(url)
                score = evaluate_ai_content_locally(scraped_text)
                raw_scores.append(score)

                if scraped_text:
                    if score is not None and score >= deep_scan_threshold:
                        # Get the list of dictionaries from our updated function
                        flagged_snippets = get_highly_likely_ai_sentences(scraped_text, threshold=0.80)

                        if flagged_snippets:
                            # 1. Format a simple string for the Streamlit UI to prevent PyArrow crashes
                            ui_preview = f"{len(flagged_snippets)} AI blocks flagged. See Excel export."
                            extracted_ai_texts_ui.append(ui_preview)

                            # 2. Append the complex dictionary data for the Excel sheet
                            for snippet in flagged_snippets:
                                master_flagged_list.append({
                                    "URL": url,
                                    "AI_Probability": snippet["score"],
                                    "Flagged_Text": snippet["text"]
                                })
                        else:
                            extracted_ai_texts_ui.append("No text blocks flagged above threshold.")
                    else:
                        extracted_ai_texts_ui.append(f"Skipped (Overall AI prob < {deep_scan_threshold_ui}%).")
                else:
                    extracted_ai_texts_ui.append("Failed to scrape.")

                progress_bar.progress((index + 1) / total_urls)

            status_text.text("✅ Evaluation complete! Grading on a curve...")

            # Convert our detailed master list into a separate DataFrame for Excel
            detailed_flagged_df = pd.DataFrame(master_flagged_list)

            # Assign ONLY the simple strings to the main DataFrame for the UI
            df['Raw_AI_Prob'] = raw_scores
            df['Flagged_AI_Text'] = extracted_ai_texts_ui

            # Fill failed scrapes with median site score
            median_score = df['Raw_AI_Prob'].median()
            df['Raw_AI_Prob'] = df['Raw_AI_Prob'].fillna(median_score)

            try:
                bins = pd.qcut(df['Raw_AI_Prob'], q=10, duplicates='drop')
                if len(bins.cat.categories) == 0:
                    df['AI_Score'] = 5
                else:
                    df['AI_Score'] = bins.cat.codes + 1
            except Exception:
                df['AI_Score'] = 5

            st.write("### Scored Data")
            # Now this will render safely because 'Flagged_AI_Text' contains only simple text strings
            st.dataframe(df[['URL', 'Click_Change', 'AI_Score', 'Raw_AI_Prob', 'Flagged_AI_Text']])

            # --- REGRESSION AND VISUALIZATION ---
            st.write("### Linear Regression Analysis")

            plot_df = df.dropna(subset=['Click_Change'])

            # Default values in case the regression cannot run
            r_squared, p_value, slope = 0.0, 0.0, 0.0

            if len(plot_df) > 1:
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

                # --- EXTRACT STATS ---
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

            else:
                st.error("Not enough valid 'Click_Change' data to perform a linear regression. Please ensure your CSV has numerical values in the Click_Change column.")

            # --- ALWAYS SHOW DOWNLOAD BUTTON ---
            st.divider()
            st.write("### Export Report")

            # Pass the detailed_flagged_df into the generator
            excel_file = generate_excel(df, detailed_flagged_df, r_squared, p_value, slope)

            st.download_button(
                label="Download Report (Excel / Google Sheets)",
                data=excel_file,
                file_name="ai_impact_analysis_with_flags.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
