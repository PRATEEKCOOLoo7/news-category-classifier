# Quick Start Guide

Get up and running with the News Category Classifier in minutes!

## 🚀 Fast Setup (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset

**Option A: Automated (Recommended)**
```bash
python download_dataset.py
```

**Option B: Manual**
1. Go to https://www.kaggle.com/datasets/rmisra/news-category-dataset
2. Download and extract to `data/` folder

### Step 3: Train Model

```bash
python train.py
```

⏱️ **Training Time:** 30-60 min (GPU) or 3-5 hours (CPU)

---

## 📊 Making Predictions

After training completes:

```bash
python predict.py --text "Your news headline here"
```

**Example:**
```bash
python predict.py --text "Biden announces new climate policy"
```

**Output:**
```
Predicted Category: POLITICS
Confidence: 87.34%
```

---

## 🔍 Verify Setup

Check everything is installed correctly:

```bash
python setup_check.py
```

---

## 📈 Generate Visualizations

After training:

```bash
python visualize.py
```

Creates:
- Training/validation curves
- Per-category performance charts
- Data distribution plots

---

## 💡 Tips

### Training Faster
- Use a GPU (40% of training time)
- Reduce `--batch_size 8` if running out of memory
- Reduce `--epochs 2` for quick testing

### Better Results
- Increase `--epochs 5`
- Try `--model_name bert-base-uncased` (slower but more accurate)
- Increase `--max_length 256` for longer texts

### Memory Issues
```bash
# Reduce batch size
python train.py --batch_size 8

# Or use gradient accumulation
python train.py --batch_size 4 --gradient_accumulation_steps 4
```

---

## 📁 Project Files

```
news-category-classifier/
├── train.py              ← Train the model
├── predict.py            ← Make predictions
├── download_dataset.py   ← Download dataset
├── setup_check.py        ← Verify installation
├── visualize.py          ← Create plots
├── requirements.txt      ← Dependencies
└── README.md             ← Full documentation
```

---

## 🆘 Troubleshooting

### "Dataset not found"
```bash
python download_dataset.py
```

### "Kaggle credentials not found"
1. Go to https://www.kaggle.com/account
2. Download API token (kaggle.json)
3. Place in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows)

### "Out of memory"
```bash
python train.py --batch_size 8
```

### "Takes too long"
- Use GPU if available
- Reduce epochs: `python train.py --epochs 2`

---

## 📝 For Submission

1. **Create GitHub repo**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: News category classifier"
   ```

2. **Push to GitHub**
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

3. **Share the link** via email/portal

---

**Need more details?** See [README.md](README.md)
