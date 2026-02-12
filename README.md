# News Category Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange.svg)](https://huggingface.co/transformers/)

A production-ready news category classification system using transformer-based models (DistilBERT) to predict article categories from headlines and short descriptions.

## 📊 Project Overview

This project implements a multi-class text classification model that categorizes news articles from the HuffPost dataset into 40+ categories including Politics, Business, Sports, Entertainment, Tech, and more.

**Key Features:**
- ✅ DistilBERT fine-tuning for efficient inference
- ✅ Comprehensive evaluation with F1-score, accuracy, and confusion matrix
- ✅ Command-line inference tool with confidence scores
- ✅ Modular, production-ready code structure
- ✅ Complete reproducibility with detailed documentation

## 📁 Project Structure

```
news-category-classifier/
├── data/                           # Dataset directory
│   └── News_Category_Dataset_v3.json
├── models/                         # Trained model artifacts
│   ├── config.json                # Model configuration
│   ├── pytorch_model.bin          # Model weights
│   ├── tokenizer files            # Tokenizer configuration
│   ├── label_encoder.npy          # Label mappings
│   ├── confusion_matrix.png       # Visualization
│   └── training_summary.json      # Training metrics
├── src/                           # Source modules
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading & preprocessing
│   ├── preprocessor.py           # Tokenization & PyTorch datasets
│   └── model.py                  # Model configuration
├── train.py                       # Main training script
├── predict.py                     # Inference script
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                      # This file
```

## 📦 Dataset

**Name:** News Category Dataset (HuffPost)

**Source:** [Kaggle - News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

**Statistics:**
- 📰 ~210,000 news articles
- 🏷️ 42 unique categories
- 📝 Headlines + short descriptions
- 📅 Published between 2012-2022

**Categories include:** Politics, Wellness, Entertainment, Travel, Style & Beauty, Parenting, Healthy Living, Queer Voices, Food & Drink, Business, Comedy, Sports, Black Voices, Home & Living, Parents, The Worldpost, Weddings, Women, Impact, Divorce, Crime, Media, Weird News, Green, Worldpost, Religion, Style, Science, World News, Taste, Tech, Money, Arts, College, Latino Voices, Culture & Arts, Fifty, Good News, Arts & Culture, Environment, Education

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd news-category-classifier
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📥 Download Dataset

### Option 1: Using Kaggle API (Recommended)

```bash
# Install Kaggle CLI (already in requirements.txt)
pip install kaggle

# Setup Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create API token (downloads kaggle.json)
# 3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d rmisra/news-category-dataset -p data --unzip
```

### Option 2: Manual Download

1. Go to [https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
2. Click "Download" button
3. Extract `News_Category_Dataset_v3.json` to the `data/` directory

## 🎓 Training

### Basic Training

```bash
python train.py
```

### Training with Custom Parameters

```bash
python train.py \
  --model_name distilbert-base-uncased \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --max_length 128
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `distilbert-base-uncased` | Pretrained model from Hugging Face |
| `--data_path` | `data/News_Category_Dataset_v3.json` | Path to dataset |
| `--output_dir` | `models` | Directory to save trained model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--max_length` | `128` | Maximum sequence length |

**Expected Training Time:**
- 🖥️ CPU: 3-5 hours
- 🚀 GPU: 30-60 minutes

## 📊 Evaluation Results

After training, the model will generate:

1. **Validation Metrics** (printed to console)
   - Accuracy
   - Macro F1-score
   - Weighted F1-score

2. **Confusion Matrix** → `models/confusion_matrix.png`

3. **Classification Report** → `models/classification_report.json`

4. **Training Summary** → `models/training_summary.json`

### Sample Results

```
Validation Metrics:
  Accuracy:      0.7234
  F1 (Macro):    0.6845
  F1 (Weighted): 0.7189
```

## 🔮 Inference

### Command Line Prediction

```bash
python predict.py --text "Biden announces new climate policy initiative"
```

### Output Example

```
==============================================================
PREDICTION RESULTS
==============================================================

Input Text:
  Biden announces new climate policy initiative

Predicted Category: POLITICS
Confidence: 87.34%

Top 3 Predictions:
  1. POLITICS            ████████████████████████████████████ 87.34%
  2. ENVIRONMENT         ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8.92%
  3. GREEN               ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 2.14%
==============================================================
```

### JSON Output

```bash
python predict.py --text "Tech giant unveils new AI chip" --json
```

```json
{
  "text": "Tech giant unveils new AI chip",
  "predicted_category": "TECH",
  "confidence": 0.9123,
  "top_predictions": [
    {"category": "TECH", "confidence": 0.9123},
    {"category": "BUSINESS", "confidence": 0.0567},
    {"category": "SCIENCE", "confidence": 0.0234}
  ]
}
```

### Multiple Predictions

```bash
# Example mode (runs predefined examples)
python predict.py
```

## 🏗️ Model Architecture

**Base Model:** `distilbert-base-uncased`
- 📐 66M parameters
- ⚡ 40% faster than BERT
- 💾 40% smaller than BERT
- 🎯 Retains 97% of BERT's performance

**Fine-tuning:**
- Classification head for 42 categories
- AdamW optimizer
- Linear warmup scheduler
- Early stopping with patience=3

## 🧪 Testing the Pipeline

```bash
# Test data loading
python -c "from src.data_loader import load_dataset; load_dataset()"

# Test tokenization
python src/preprocessor.py

# Test model creation
python src/model.py
```

## 📈 Bonus Features

### Confusion Matrix Visualization

Automatically generated during training and saved to `models/confusion_matrix.png`

### Per-Category Performance

Detailed F1-scores for each category in `models/classification_report.json`

## 🛠️ Dependencies

- **transformers** - Hugging Face transformers library
- **torch** - PyTorch deep learning framework
- **scikit-learn** - ML utilities and metrics
- **pandas** - Data manipulation
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization

See [`requirements.txt`](requirements.txt) for complete list with versions.

## 📝 Code Quality

- ✅ Modular architecture with separate concerns
- ✅ Comprehensive docstrings
- ✅ Type hints where applicable
- ✅ Proper error handling
- ✅ Logging and progress tracking
- ✅ Reproducible results (fixed random seeds)

## 🔄 Reproducibility

All experiments are reproducible with:
- Fixed random seeds (42)
- Stratified train/test splits
- Deterministic training configuration
- Saved model checkpoints and configurations

## 📜 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created as part of a machine learning assessment demonstrating:
- Production-ready ML code development
- Transformer fine-tuning expertise
- End-to-end ML pipeline implementation

## 🙏 Acknowledgments

- **Dataset:** Rishabh Misra (HuffPost News Category Dataset)
- **Model:** Hugging Face (DistilBERT)
- **Framework:** PyTorch & Transformers

---

**For questions or issues, please open a GitHub issue.**
