import streamlit as st
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page title
st.title("Fake Review Detector")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_fake_reviews.csv")
    df['cleaned'] = df['review'].apply(clean_text)
    return df

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load and process data
df = load_data()

# Vectorize and train model
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'genuine': 0, 'fake': 1})

model = LogisticRegression()
model.fit(X, y)

# UI: User input
review_input = st.text_area("Enter a product review to check if it's fake or genuine:")

if st.button("Check Review"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("This review is likely **FAKE**.")
        else:
            st.success("This review seems **GENUINE**.")
