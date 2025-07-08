import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE

# Set nltk path if needed
nltk.data.path.append(r"C:/Users/shonu/nltk_data")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w.isalnum()]
    text = [w for w in text if w not in stopwords.words('english')]
    text = [ps.stem(w) for w in text]
    return " ".join(text)

# Load data
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['transformed'] = df['text'].apply(transform_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed']).toarray()
y = df['label']

# Balance data
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=2)

# Train model
svc = LinearSVC()
svc.fit(X_train, y_train)

# Evaluate
y_pred = svc.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(svc, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
print("✅ Model and vectorizer saved.")
