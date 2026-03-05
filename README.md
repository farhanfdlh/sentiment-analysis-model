# 🤖 Sentiment Analysis — Chatbot Reviews (SVM + SMOTE)

This project implements **sentiment analysis** on user reviews of an AI chatbot (DeepSeek) using a **Support Vector Machine (SVM)** classifier, with class imbalance handled through **SMOTE** (Synthetic Minority Oversampling Technique).

---

## 📌 Project Overview

The goal of this project is to classify user reviews into **Positive** or **Negative** sentiment categories. The full pipeline covers data scraping, Indonesian text preprocessing, lexicon-based labeling, model training, and evaluation.

---

## 🗂️ Repository Structure

```
sentiment-analysis-model/
├── data/
│   ├── raw/
│   │   ├── scrapped_data_deepseek.xlsx       # Raw scraped review dataset
│   │   └── test_data_baru_deepseek.xlsx      # 20 new test reviews
│   ├── processed/
│   │   ├── data_preprocessing.xlsx           # Data after full preprocessing
│   │   └── data_label_pos_neg.xlsx           # Data after labeling
│   └── lexicon/
│       ├── positive_lex.tsv                  # Positive sentiment lexicon
│       ├── negative_lex.tsv                  # Negative sentiment lexicon
│       └── kamuskatabaku.xlsx                # Indonesian slang normalization dictionary
├── models/
│   ├── svm_smote_model.joblib                # Trained SVM + SMOTE model
│   └── tfidf_vectorizer.joblib               # Fitted TF-IDF Vectorizer
├── notebooks/
│   └── Chatbot_SentimentAnalysis_SVM_SMOTE.ipynb
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw Data → Preprocessing → Labeling → Modeling → Evaluation
```

### 1. 📥 Data Collection
- User reviews of the DeepSeek AI chatbot were collected via scraping

### 2. 🧹 Text Preprocessing (Indonesian Language)
| Step | Description |
|---|---|
| Sentence Splitting | Splits reviews into individual sentences using spaCy |
| Cleaning | Removes emojis, special characters, and noise |
| Case Folding | Converts all text to lowercase |
| Normalization | Replaces non-standard words using `kamuskatabaku.xlsx` |
| Tokenizing | Splits sentences into word tokens |
| Stopword Removal | Removes common uninformative words (NLTK Indonesian) |
| Stemming | Reduces words to their base form using **Sastrawi** |

### 3. 🏷️ Labeling — Lexicon Based
- Sentiment is determined by word weights from `positive_lex.tsv` and `negative_lex.tsv`
- Labels: **Positive**, **Negative** *(Neutral labels are excluded)*

### 4. 🤖 Modeling
- **Algorithm**: Linear SVC (Support Vector Machine)
- **Features**: TF-IDF Vectorization
- **Class imbalance handling**: SMOTE
- **Evaluation strategy**: 10-Fold Stratified Cross Validation

### 5. 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- G-Mean
- Confusion Matrix per fold

---

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn
pip install Sastrawi nltk spacy wordcloud matplotlib seaborn joblib
python -m spacy download xx_ent_wiki_sm
```

---

## 🚀 How to Use the Model

```python
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/svm_smote_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Predict sentiment for new text (preprocessed)
text = ["this app is very helpful and easy to use"]
text_vec = vectorizer.transform(text)
prediction = model.predict(text_vec)

print("Predicted sentiment:", prediction[0])
```

> ⚠️ **Note**: Input text should go through the full preprocessing pipeline (cleaning, normalization, stemming) before prediction for best results.

---

## 🌐 Streamlit Web App

An interactive web application for sentiment analysis is available at:

👉 **[github.com/farhanfdlh/sentiment-analysis-app](https://github.com/farhanfdlh/sentiment-analysis-app)**

---

## 👤 Author

**Farhan Fadhilah Rasyid**
- GitHub: [@farhanfdlh](https://github.com/farhanfdlh)
