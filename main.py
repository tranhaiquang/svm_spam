import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tf-idf-vectorizer.pkl")


def predict_text(text):
    # Transform the input text using the pre-trained vectorizer and scaler
    input_vector = vectorizer.transform([text])
    scaled_input = input_vector.toarray()
    
    # Make prediction
    prediction = svm_model.predict(scaled_input)

    # Get the probability prediction
    probability_scores = svm_model.predict_proba(scaled_input)
    spam_probability = probability_scores[0][1]

    if prediction[0] == "ham":
        return "not spam", round(probability_scores[0][0] * 100, 2)
    
    return prediction[0], round(probability_scores[0][1] * 100, 2)

def get_words(text):
    # Split the input text by whitespace to get individual words
    words = text.split()
    return len(words)

st.title('Spam Detector')
st.text('Copy/paste an email message to detect')

input_text = st.text_area("Title", label_visibility="collapsed")

if st.button("Check", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        
        # Display overall score
        st.markdown("### Analysis Summary")
        st.write(f"Spam: :blue[{predict_text(input_text)[0]}]")
        st.write(f"Probability Score: :blue[{predict_text(input_text)[1]}%]")
        st.write("Words:", get_words(input_text))
        
        
    


