"""
Quick fix to regenerate label_encoder.npy with current NumPy version
"""
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = []
with open('data/News_Category_Dataset_v3.json', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(df['category'])

# Save with compatible format
np.save('models/label_encoder.npy', label_encoder.classes_)

print(f"✓ Saved {len(label_encoder.classes_)} categories")
print("Categories:", label_encoder.classes_[:10], "...")
