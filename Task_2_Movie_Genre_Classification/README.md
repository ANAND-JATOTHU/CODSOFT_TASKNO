# Task 2 – Movie Genre Classification 🎬

> **CodSoft AI/ML Internship | Task 2**

Create a machine learning model that predicts the genre of a movie based on its plot summary. This project implements a **TF-IDF Vectorizer** combined with **Logistic Regression** and **Naive Bayes** classifiers to accurately categorize movie descriptions.

---

## 📌 Google Colab Optimization

This project is optimized to run on **Google Colab**.
- 💾 **Google Drive Integration**: Automatically mounts your Google Drive to securely save your trained models (logistic_regression_model.pkl and 	fidf_vectorizer.pkl).
- 📥 **Auto-Download**: Seamlessly downloads the dataset from Kaggle straight into the Colab environment using your Kaggle API token.

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. **Upload** movie_genre_classification.ipynb
3. **Run All Cells**: 
   - When prompted, **Connect to Google Drive** (this creates a folder CodSoft_Task2 in your Drive to save models).
   - In Step 2, you will be prompted to enter your Kaggle API Token (starts with KGAT_...). You can generate this in your Kaggle Account Settings under API.
4. Let the notebook train! The final .pkl model files will be safely stored in your Google Drive under MyDrive/CodSoft_Task2/models/.

---

## 🧠 Model Architecture & Pipeline

1. **Text Preprocessing**: 
   - Lowercasing text
   - Removing punctuation and special characters
   - Removing English stop words
   - Stemming words using PorterStemmer
2. **Feature Extraction**: 
   - TfidfVectorizer mapping text to 10,000 TF-IDF features, using unigrams and bigrams.
3. **Classifiers**:
   - **Logistic Regression**: Used as the primary model due to its high performance on text classification.
   - **Multinomial Naive Bayes**: Used as a baseline model.
4. **Evaluation**:
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-Score)
