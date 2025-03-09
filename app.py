import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re
from io import BytesIO
import os

# Custom CSS for Dark Mode
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.5rem;
        color: #1e90ff;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.5rem;
        color: #1e90ff;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-size: 18px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #2c3e50;
        color: #e0e0e0;
        font-size: 16px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .download-button {
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load pre-trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r'\W', ' ', text)
    return text

# App Title
st.markdown("<div class='title'>IMDB Review Sentiment Analysis</div>", unsafe_allow_html=True)
st.write("Enter a movie review below, and the model will predict if it's positive or negative.")

# Single review input
user_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
        color = "green" if prediction == 1 else "red"
        st.markdown(
            f"""
            <div class="prediction-box" style="background-color: {color};">
                Sentiment: <strong>{sentiment}</strong>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a review first.")

# File uploader section for bulk analysis
st.markdown("<div class='section-title'>Bulk Review Analysis</div>", unsafe_allow_html=True)
st.write('<div style="color:#FF003F;">For multiple reviews, upload a CSV file containing a column named "review".</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file with reviews", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "review" not in df.columns:
        st.error("CSV file must contain a 'review' column.")
    else:
        st.success("File uploaded successfully!")
        
        # Use the uploaded file's name to create output file names
        base_name = os.path.splitext(uploaded_file.name)[0]
        csv_filename = f"{base_name}_labeled.csv"
        plot_filename = f"{base_name}_sentiment_distribution.png"
        
        # Preprocess reviews and predict sentiments
        df["processed_review"] = df["review"].apply(preprocess_text)
        X = vectorizer.transform(df["processed_review"])
        df["sentiment"] = model.predict(X)
        df["sentiment"] = df["sentiment"].replace({1: "Positive", 0: "Negative"})
        
        st.write("### Preview of Labeled Reviews:")
        st.write(df.head())
        
        # Calculate and display sentiment percentages and counts
        total_reviews = len(df)
        positive_count = (df["sentiment"] == "Positive").sum()
        negative_count = (df["sentiment"] == "Negative").sum()
        positive_percentage = (positive_count / total_reviews) * 100
        negative_percentage = (negative_count / total_reviews) * 100

        st.markdown(
            f"""
            <div class="info-box">
                <b>Positive Reviews:</b> {positive_percentage:.2f}%<br>
                <b>Negative Reviews:</b> {negative_percentage:.2f}%<br>
                <b>Counts:</b> Positive = {positive_count} | Negative = {negative_count}
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Download labeled reviews CSV using dynamic filename
        st.download_button(
            label="Download Labeled Reviews", 
            data=df.to_csv(index=False), 
            file_name=csv_filename, 
            mime='text/csv', 
            key='csv_download'
        )
        
        # Generate sentiment distribution plot
        sentiment_counts = df["sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["red", "green"] if "Negative" in sentiment_counts.index and "Positive" in sentiment_counts.index else ["blue", "orange"]
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
        
        # Save plot to buffer for download using dynamic filename
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        
        st.download_button(
            label="Download Sentiment Distribution Plot",
            data=buf,
            file_name=plot_filename,
            mime="image/png",
            key='plot_download'
        )
