# Dataset Download Guide

## Option 1: Manual Download (Easiest - 5 minutes)

### Step-by-Step:

1. **Go to the Kaggle dataset page:**
   - Open: https://www.kaggle.com/datasets/rmisra/news-category-dataset
   - (You may need to create a free Kaggle account if you don't have one)

2. **Download the dataset:**
   - Click the **"Download"** button (top right)
   - This will download `archive.zip` (about 60 MB)

3. **Extract the file:**
   - Unzip `archive.zip`
   - You'll find `News_Category_Dataset_v3.json`

4. **Move the file:**
   - Copy `News_Category_Dataset_v3.json` to:
   ```
   c:\Users\Prateek Srivastava\OneDrive - Tuck School of Business at Dartmouth College\Desktop\MLA\news-category-classifier\data\
   ```

5. **Verify:**
   ```bash
   python setup_check.py
   ```

**That's it!** You're ready to train.

---

## Option 2: Kaggle API (Automated - 10 minutes setup)

### Step-by-Step:

1. **Get your Kaggle API credentials:**
   - Go to: https://www.kaggle.com/account
   - Scroll to "API" section
   - Click **"Create New API Token"**
   - This downloads `kaggle.json`

2. **Place the credentials file:**
   - Create directory: `C:\Users\Prateek Srivastava\.kaggle\`
   - Move `kaggle.json` to that directory

3. **Download automatically:**
   ```bash
   python download_dataset.py
   ```

---

## Quick Check

After downloading, verify the file exists:

```bash
dir data\News_Category_Dataset_v3.json
```

Should show a ~60 MB JSON file.

---

## Next Step After Download

Once the dataset is in place, you're ready to train:

```bash
python train.py
```

This will take 30-60 minutes on GPU or 3-5 hours on CPU.
