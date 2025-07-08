from flask import Flask, render_template, request
import pickle
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Set path to nltk data
nltk.data.path = [r"C:/Users/shonu/nltk_data"]

# Load the newly trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmed = [ps.stem(word) for word in tokens]
    return " ".join(stemmed)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = request.form['message']
        transformed = transform_text(message)
        vec = vectorizer.transform([transformed]).toarray()
        prediction = model.predict(vec)[0]
        result = "Spam ❌" if prediction == 1 else "Not Spam ✅"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
