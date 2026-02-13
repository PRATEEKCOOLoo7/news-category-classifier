# Google Colab Training Guide

## Quick Setup (5 minutes)

### Step 1: Enable GPU in Colab
1. In Colab, click: **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

### Step 2: Upload the Notebook
1. In Colab's "Open notebook" dialog, click **Upload**
2. Upload `train_colab.ipynb` from your project folder

### Step 3: Run Training
1. Click **Runtime** → **Run all** (or run cells one by one)
2. When prompted, upload `News_Category_Dataset_v3.json`
3. Training will start automatically

**Training time with GPU: 30-45 minutes** ⚡

### Step 4: Download Trained Model
- After training completes, `models.zip` will auto-download
- Extract and place in your local project's `models/` directory

---

## Alternative: Manual Upload

If you prefer to manually set up in Colab:

1. Create new notebook in Colab
2. Enable GPU (Runtime → Change runtime type → T4 GPU)
3. Copy-paste code cells from `train_colab.ipynb`
4. Run all cells

---

## After Training

1. Download `models.zip` from Colab
2. Extract to your local project
3. Test locally:
   ```bash
   python predict.py --text "Your headline here"
   ```
4. Push to GitHub and submit!

---

## Troubleshooting

**"GPU not available"**
- Make sure you selected T4 GPU in runtime settings
- Free Colab has usage limits; try again later if exceeded

**"Out of memory"**
- Reduce batch size to 8 in the training cell
- Or restart runtime and try again

**Upload fails**
- Alternative: Upload dataset to Google Drive, then mount drive in Colab
