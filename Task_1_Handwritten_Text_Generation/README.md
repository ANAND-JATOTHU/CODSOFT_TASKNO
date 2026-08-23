# Task 1 – Handwritten Text Generation 🖊️

> **CodSoft AI/ML Internship | Task 1**

Implement a **character-level Recurrent Neural Network (RNN)** to generate handwritten-like text. The model is trained on the [English Handwritten Characters Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset).

---

## 📌 Google Colab Optimization

This project has been updated to be run **exclusively on Google Colab**.

Key features of this Colab version:
- 💾 **Google Drive Integration**: Automatically mounts your Google Drive to save the trained model (char_rnn_best.h5 and char_rnn_final.h5) and outputs so they are preserved after the session ends.
- 🚀 **T4 GPU Support**: Ready to be executed on Colab's free T4 GPU backend for fast training.
- 📥 **Auto-Download**: Automatically downloads the dataset directly into the Colab environment.

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. **Upload** handwritten_text_generation.ipynb
3. **Change Runtime**: Go to Runtime -> Change runtime type -> select **T4 GPU**.
4. **Run All Cells**: 
   - When prompted, **Connect to Google Drive** (this creates a folder CodSoft_Task1 in your Drive to save models).
   - In Step 2, you will be prompted to upload your kaggle.json file. (You can generate this in your Kaggle Account Settings under API).
5. Let the notebook train! The final .h5 model files and generated image outputs will be safely stored in your Google Drive under MyDrive/CodSoft_Task1/.

---

## 🧠 Model Architecture

The character-level LSTM pipeline uses:
- **Embedding Layer** (Vocab Size → 64 dims)
- **LSTM Layer 1** (256 units) + Dropout
- **LSTM Layer 2** (256 units) + Dropout
- **LSTM Layer 3** (128 units) + Dropout
- **Dense Layer** (128 units, ReLU) + BatchNorm
- **Output Layer** (Vocab Size, Softmax)

It utilizes **Temperature Sampling** to control the creativity (diversity) of the generated text sequences, and renders them visually by stitching together actual handwritten character images.
