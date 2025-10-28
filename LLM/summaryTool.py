import streamlit as st
from transformers import pipeline
import time

# ----------------------------
# 🔹 App Title
# ----------------------------
st.set_page_config(page_title="Summarizer")
st.title("Text Summarizer")

st.markdown("Enter text, choose your models, and get both a summary and sentiment instantly!")

# ----------------------------
# 🔹 Model Selection Section
# ----------------------------
st.subheader("Model Settings")

# Summarization model options
sum_model = st.selectbox(
    "Choose Summarization Model:",
    [
        "facebook/bart-large-cnn",
        "google/pegasus-xsum",
        "t5-base",
        "sshleifer/distilbart-cnn-12-6"
    ],
    index=0
)

# Sentiment model options
sent_model = st.selectbox(
    "Choose Sentiment Analysis Model:",
    [
        "distilbert-base-uncased-finetuned-sst-2-english", 
        "nlptown/bert-base-multilingual-uncased-sentiment"
    ],
    index=0
)

# ----------------------------
# 🔹 User Input Section
# ----------------------------
text = st.text_area("Paste your text here:", height=200)

# Summary length options
summary_length = st.slider("Select summary length (approximate words):", 30, 300, 100, step=10)

# ----------------------------
# 🔹 Cached Model Loaders
# ----------------------------
@st.cache_resource
def load_summarizer(model_name):
    return pipeline("summarization", model=model_name)

@st.cache_resource
def load_sentiment(model_name):
    return pipeline("sentiment-analysis", model=model_name)

# ----------------------------
# 🔹 Run Analysis Button
# ----------------------------
if st.button("Analyze Text"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner(f"Loading models and analyzing..."):
            start_time = time.time() 
            summarizer = load_summarizer(sum_model)
            sentiment_analyzer = load_sentiment(sent_model)

            # Summarization
            summary = summarizer(text, max_length=summary_length, min_length=int(summary_length/2), do_sample=False)[0]['summary_text']

            # Sentiment
            sentiment_result = sentiment_analyzer(text)[0]

            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time

        # ----------------------------
        # 🔹 Display Results
        # ----------------------------
        st.success(f"Analysis Complete in {elapsed_time:.2f} seconds!")

        st.subheader("Summary:")
        st.write(summary)

        st.subheader("Sentiment Analysis:")
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        if "NEG" in sentiment_label.upper():
            color = "🔴 Negative"
        elif "POS" in sentiment_label.upper():
            color = "🟢 Positive"
        else:
            color = "🟡 Neutral"

        st.write(f"**Sentiment:** {color}")
        st.write(f"**Confidence:** {(sentiment_score * 100):.2f}%")
