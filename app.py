from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# NLP setup
nltk.download('stopwords')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_v = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Preprocess function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('news')  # Get JSON data from POST request
    processed_text = preprocess_text(data)
    vectorized_text = tfidf_v.transform([processed_text])
    prediction = model.predict(vectorized_text)
    result = "Real News 📰" if prediction[0] == 1 else "Fake News ⚠️"
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
