# spam-classifier
#  Spam Message Classifier using Flask & Machine Learning

This project is a simple **Spam Detection Web App** that allows users to input a message and predicts whether it is **Spam ❌** or **Not Spam ✅** using a trained ML model.

---

##  Features

- Clean & Simple Flask Web Interface
- Preprocessing using **NLTK** (tokenization, stopwords removal, stemming)
- Machine Learning models (LinearSVC, Naive Bayes, etc.)
- Trained using **TF-IDF** vectorizer
- Supports deployment via **local server**
- Custom spam dataset support

---

##  Model Details

- **Vectorizer**: TfidfVectorizer
- **Classifier**: LinearSVC
- **Metrics**:
  - Accuracy: `99.83%`
  - Precision: `99.88%`
  - Recall: `99.77%`
  - F1 Score: `99.83%`

---

## Folder Structure
spam-detector/ 
  app.py 
  train_model.py 
  test_nltk.py 
  model_new.pkl 
  vectorizer_new.pkl 
  spam.csv 
  templates/ 
    index.html 
  static/ 
    style.css 


