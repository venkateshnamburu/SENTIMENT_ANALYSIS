import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Sentiment Analysis Web App")
st.write("Enter a review below to analyze its sentiment.")

user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Sentiment: **{sentiment}**")
    else:
        st.write("Please enter a review.")
