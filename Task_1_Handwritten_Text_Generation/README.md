# 🖊️ Task 1: Handwritten Text Generation
### CodSoft AI/ML Internship — Character-Level RNN (LSTM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX%203050%20%7C%20T4-green)](https://www.nvidia.com/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset)

---

## 📖 Project Description

This project implements a **Character-Level Recurrent Neural Network (RNN)** using stacked LSTM layers to generate handwritten-like text sequences. The model learns the sequential patterns of characters from the **English Handwritten Characters Dataset** and generates new character sequences at multiple creativity temperatures.

### What it does:
1. **Trains** a stacked 3-layer LSTM on character label sequences from the dataset
2. **Generates** new character sequences using temperature-controlled sampling
3. **Renders** generated text as real handwritten images by stitching actual dataset images
4. **Visualises** training curves, EDA plots, and character frequency analysis

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| **Name** | English Handwritten Characters Dataset |
| **Source** | [Kaggle – dhruvildave](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset) |
| **Classes** | 62 (0–9, A–Z, a–z) |
| **Images per class** | ~55 |
| **Total images** | ~3,410 |
| **Format** | PNG images + `english.csv` label file |

### Dataset structure expected locally:
```
Task_1_Handwritten_Text_Generation/
└── dataset/
    ├── english.csv        ← image filename → label mapping
    └── Img/
        ├── img001-001.png
        ├── img001-002.png
        └── ...
```

---

## 🚀 How to Run

### Option A — Local Machine (RTX 3050 6 GB)

```bash
# 1. Navigate to project folder
cd "Task_1_Handwritten_Text_Generation"

# 2. (First time) Download dataset from Kaggle
#    Place your kaggle.json at ~/.kaggle/kaggle.json, then:
kaggle datasets download -d dhruvildave/english-handwritten-characters-dataset -p dataset --unzip

# 3. Generate the notebook
python create_notebook.py

# 4. Launch Jupyter
jupyter notebook handwritten_text_generation.ipynb
```

> **Tip:** The notebook auto-detects your GPU and enables `set_memory_growth` to prevent OOM errors on the RTX 3050 6 GB.

---

### Option B — Google Colab (T4 15 GB)

1. Upload `handwritten_text_generation.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → GPU (T4)**
3. Run **Cell 2** (environment detection) — it will detect Colab automatically
4. Run **Cell 4** — upload `kaggle.json` when prompted
5. Run all remaining cells

---

### Option C — Kaggle Notebook

1. Create a new Kaggle Notebook
2. Add dataset: `dhruvildave/english-handwritten-characters-dataset`
3. Enable **GPU Accelerator** in notebook settings
4. Upload `handwritten_text_generation.ipynb` and run all cells
5. Cell 2 detects Kaggle and sets `DATA_DIR` automatically — no download needed

---

## 🧠 Model Architecture

| Layer | Type | Units / Output | Parameters |
|-------|------|----------------|------------|
| Embedding | `Embedding` | 62 → 64-dim | ~3,968 |
| LSTM 1 | `LSTM` | 256 units, `return_sequences=True` | ~328,704 |
| Dropout 1 | `Dropout` | rate=0.3 | — |
| LSTM 2 | `LSTM` | 256 units, `return_sequences=True` | ~525,312 |
| Dropout 2 | `Dropout` | rate=0.3 | — |
| LSTM 3 | `LSTM` | 128 units, `return_sequences=False` | ~197,120 |
| Dropout 3 | `Dropout` | rate=0.2 | — |
| Dense Hidden | `Dense` + ReLU | 128 | ~16,512 |
| Batch Norm | `BatchNormalization` | — | ~512 |
| Output | `Dense` + Softmax | 62 | ~7,998 |

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Cross-Entropy
- **Total Parameters:** ~1.08 M
- **Estimated Size:** ~4.3 MB

---

## 🌡️ Temperature Sampling

Temperature controls the **creativity vs. consistency** of generated text:

| Temperature | Behaviour | Use Case |
|-------------|-----------|----------|
| `T = 0.5` | **Conservative** — repeats frequent patterns | Predictable, structured output |
| `T = 0.8` | **Balanced** — natural variation | General-purpose generation |
| `T = 1.0` | **Neutral** — raw model distribution | Standard sampling |
| `T = 1.2` | **Creative** — more character diversity | Exploration |
| `T = 1.5` | **Very creative** — highly random | Maximum diversity |

**Formula:**
```
p_i = exp(log(p_i) / T) / Σ exp(log(p_j) / T)
```

---

## ⚙️ Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 40 |
| Step size | 3 |
| Batch size (GPU) | 256 |
| Batch size (CPU) | 64 |
| Max epochs | 50 |
| Embedding dim | 64 |
| LSTM units | 256 |
| Early stopping patience | 8 |
| LR reduction patience | 4 |
| Min learning rate | 1e-6 |

---

## 📁 File Structure

```
Task_1_Handwritten_Text_Generation/
│
├── handwritten_text_generation.ipynb   ← Main notebook (42 cells)
├── create_notebook.py                  ← Script that generates the notebook
├── README.md                           ← This file
├── .gitignore
│
├── dataset/                            ← Downloaded dataset (gitignored)
│   ├── english.csv
│   └── Img/
│
├── models/                             ← Saved models (*.h5 gitignored)
│   ├── char_rnn_best.h5
│   ├── char_rnn_final.h5
│   └── vocab.json
│
└── outputs/                            ← Generated plots and text
    ├── eda_distribution.png
    ├── sample_characters.png
    ├── training_curves.png
    ├── generated_handwritten_text.png
    ├── char_frequency_analysis.png
    └── generated_text.txt
```

---

## 📦 Requirements

```
tensorflow>=2.10
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
opencv-python>=4.5
Pillow>=8.0
scikit-learn>=0.24
tqdm>=4.62
kaggle>=1.5        # for dataset download
jupyter>=1.0
```

Install all at once:
```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python Pillow scikit-learn tqdm kaggle jupyter
```

---

## 📊 Notebook Cell Map

| Cell | Type | Description |
|------|------|-------------|
| 0 | Markdown | Title |
| 1–2 | MD + Code | Environment Detection |
| 3–4 | MD + Code | Dataset Download |
| 5–6 | MD + Code | Install Dependencies |
| 7–8 | MD + Code | Import Libraries |
| 9–10 | MD + Code | Load Dataset |
| 11–12 | MD + Code | EDA |
| 13–14 | MD + Code | Sample Images |
| 15–16 | MD + Code | Image Bank |
| 17–18 | MD + Code | Vocabulary |
| 19–20 | MD + Code | Training Sequences |
| 21–22 | MD + Code | Model Building |
| 23–24 | MD + Code | Callbacks |
| 25–26 | MD + Code | Training |
| 27–28 | MD + Code | Training Curves |
| 29–30 | MD + Code | Temperature Sampling |
| 31–32 | MD + Code | Text Generation |
| 33–34 | MD + Code | Image Rendering |
| 35–36 | MD + Code | Frequency Analysis |
| 37–38 | MD + Code | Interactive Generation |
| 39–40 | MD + Code | Save Model |
| 41 | Markdown | Summary Table |

---

## 👤 Author

**Jatothu Anand**  
CodSoft AI/ML Internship — Task 1  
GitHub: [@ANAND-JATOTHU](https://github.com/ANAND-JATOTHU)

---

> *"Generating handwriting, one character at a time."* 🖊️
