import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess input text
def preprocess_text(text):
    # Perform any preprocessing steps needed for your specific model
    processed_text = text  # Placeholder for actual preprocessing
    return processed_text

def predict_comment_type(input_text):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)

    # Vectorize the processed input using the loaded TF-IDF vectorizer
    input_vector = tfidf_vectorizer.transform([processed_text])

    # Make predictions using the SVM model
    prediction = svm_model.predict(input_vector)

    return prediction[0]  # Return the predicted comment type

def main():
    st.title('Comment Type Prediction')

    # Text input for user
    user_input = st.text_area("Enter the comment you want to analyze:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Get the prediction for the user input
            prediction = predict_comment_type(user_input)

            # Interpret the prediction result
            if prediction == 'true':
                st.success("Prediction: True (Genuine Comment)")
            elif prediction == 'fake':
                st.error("Prediction: Fake (Misleading Comment)")
            else:
                st.info("Prediction: Uncertain")

if __name__ == '__main__':
    main()
