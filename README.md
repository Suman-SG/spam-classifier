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
  requirements.txt (flask
                    nltk
                    scikit-learn
                    pandas
                    numpy)

    
---

##  How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-message-classifier.git
   cd spam-message-classifier
   (optional )=create a virtual environment
   pip install -r requirements.txt
   python download_nltk_data.py
   python app.py
   http://127.0.0.1:5000/
2.Requirements:
              flask
              nltk
              scikit-learn
              pandas
              numpy
3.Credits
Dataset: SMS Spam Collection Dataset

Author: suman ghosh







