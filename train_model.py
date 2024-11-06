import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')  # Ensure the punkt tokenizer is downloaded

# NLP setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load and preprocess data
train_df = pd.read_csv('train.csv')  # Ensure this file exists in your directory

# Print the columns of the dataset to check if 'Statement' column exists
print("Columns in the dataset:", train_df.columns)

# Preprocess function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Remove special chars, lowercase
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'Statement' column
train_df['processed_text'] = train_df['Statement'].apply(preprocess_text)

# Initialize TfidfVectorizer
tfidf_v = TfidfVectorizer(max_df=0.7)
X_train = tfidf_v.fit_transform(train_df['processed_text'])
y_train = train_df['Label']  # Assuming the label column is 'Label'

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Save the model and vectorizer to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_v, vectorizer_file)

print("Model training complete and saved as 'model.pkl' and 'vectorizer.pkl'")
