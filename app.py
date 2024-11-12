from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# Set up NLTK
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# NLP setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_v = pickle.load(vectorizer_file)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

# Initialize Flask app
app = Flask(__name__)

# Preprocess function
def preprocess_text(text):
    try:
        # Convert to lowercase and remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Lemmatize and remove stop words
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route with error logging
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('news')  # Get JSON data from POST request
        if not data:
            return jsonify(error="No input text provided"), 400
        
        processed_text = preprocess_text(data)
        if processed_text is None:
            return jsonify(error="Error processing text"), 500

        vectorized_text = tfidf_v.transform([processed_text])
        prediction = model.predict(vectorized_text)
        result = "Real News 📰" if prediction[0] == 1 else "Fake News ⚠️"
        return jsonify(result=result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify(error=str(e)), 500  # Return error message in case of failure

if __name__ == '__main__':
    app.run(debug=True)
