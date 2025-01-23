import streamlit as st
from naive_bayes_NLP import Process

def main():
    st.title("Amazon Review Sentiment Analysis")

    # User input for review
    user_input = st.text_area("Enter your review:")

    if st.button("Train Model"):
        # Call Process function to train the model
        Process('train')
        st.success("Model trained successfully!")

    if st.button("Predict"):
        if user_input:
            # Call Process function for prediction
            prediction = Process('predict', user_input)

            # Display prediction
            if prediction == 1:
                st.success("Positive Sentiment")
            else:
                st.error("Negative Sentiment")
        else:
            st.warning("Please enter a review.")

if __name__ == "__main__":
    main()
