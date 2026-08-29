# Task 3 – SMS Spam Classification 📱

> **CodSoft AI/ML Internship | Task 3**

Build an AI model that can classify SMS messages as spam or legitimate (ham). This project uses **TF-IDF Vectorization** alongside **Naive Bayes** and **Support Vector Machine (SVM)** classifiers. 

**Bonus Feature**: Includes a live interactive Web UI using **Gradio**!

---

## 📌 Google Colab Optimization

This project is optimized for **Google Colab**.
- 💾 **Google Drive Integration**: Automatically saves your trained models (`spam_classifier_nb.pkl` and `tfidf_vectorizer.pkl`) directly to your Drive.
- 📥 **Kaggle API Integration**: Seamlessly downloads the `uciml/sms-spam-collection-dataset` using your `KGAT_...` API token.
- 🌐 **Interactive Web UI**: Generates a public shareable web application directly from the Colab notebook.

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. **Upload** `sms_spam_detection.ipynb`
3. **Run All Cells**: 
   - Step 1 will prompt you to connect to Google Drive.
   - Step 2 will prompt you for your Kaggle API Token.
4. Scroll to the bottom (Step 11). A beautiful **Gradio Web UI** will appear right inside the notebook! You can type SMS messages and instantly test if they are Spam or Ham.

---

## 🧠 Model Architecture & Pipeline

1. **Exploratory Data Analysis**:
   - Visualizing class distribution (Spam vs. Ham).
   - Generating **Word Clouds** to see the most frequent terms in Spam vs Ham messages.
2. **Text Preprocessing**: 
   - Lowercasing, removing punctuation and numbers.
   - Removing NLTK stop words.
   - Stemming words via `PorterStemmer`.
3. **Feature Extraction**: 
   - `TfidfVectorizer` (Max features: 5000) converts text documents to a matrix of TF-IDF features.
4. **Classifiers**:
   - **Multinomial Naive Bayes**: (Best performance on text classification).
   - **Support Vector Machine (SVM)**: (Linear Kernel).
5. **Evaluation**:
   - Accuracy, Precision, Recall, F1-Score.
   - Confusion Matrix visualization using Seaborn.
