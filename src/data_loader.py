"""
Data loading and preprocessing module for HuffPost News Category Dataset.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


class NewsDataLoader:
    """Handles loading and preprocessing of the HuffPost dataset."""
    
    def __init__(self, data_path="data/News_Category_Dataset_v3.json"):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the dataset JSON file
        """
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.df = None
        self.num_classes = None
        
    def load_data(self):
        """Load the dataset from JSON file."""
        print(f"Loading dataset from {self.data_path}...")
        
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} records")
        
        return self.df
    
    def preprocess(self):
        """Preprocess the data: combine text fields and encode labels."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Combine headline and short description
        print("Combining headline and short_description...")
        self.df['text'] = self.df['headline'] + " " + self.df['short_description']
        
        # Remove any rows with missing text
        self.df = self.df.dropna(subset=['text', 'category'])
        
        # Encode category labels
        print("Encoding category labels...")
        self.df['label'] = self.label_encoder.fit_transform(self.df['category'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Number of categories: {self.num_classes}")
        print(f"Categories: {list(self.label_encoder.classes_)}")
        
        # Show category distribution
        print("\nCategory distribution:")
        print(self.df['category'].value_counts().head(10))
        
        return self.df
    
    def create_splits(self, test_size=0.2, random_state=42):
        """
        Create train/validation splits.
        
        Args:
            test_size (float): Proportion of data for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df)
        """
        if self.df is None or 'label' not in self.df.columns:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        print(f"\nCreating train/validation split ({int((1-test_size)*100)}/{int(test_size*100)})...")
        
        train_df, val_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['label']  # Maintain class distribution
        )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        return train_df, val_df
    
    def get_label_mapping(self):
        """Get the mapping between labels and category names."""
        if self.label_encoder is None:
            raise ValueError("Labels not encoded yet.")
        
        return {
            idx: label 
            for idx, label in enumerate(self.label_encoder.classes_)
        }
    
    def save_label_encoder(self, path="models/label_encoder.npy"):
        """Save the label encoder for later use."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, self.label_encoder.classes_)
        print(f"Label encoder saved to {path}")


def load_dataset(data_path="data/News_Category_Dataset_v3.json"):
    """
    Convenience function to load and preprocess the dataset.
    
    Args:
        data_path (str): Path to the dataset JSON file
        
    Returns:
        tuple: (train_df, val_df, label_encoder, num_classes)
    """
    loader = NewsDataLoader(data_path)
    loader.load_data()
    loader.preprocess()
    train_df, val_df = loader.create_splits()
    loader.save_label_encoder()
    
    return train_df, val_df, loader.label_encoder, loader.num_classes


if __name__ == "__main__":
    # Test the data loader
    train_df, val_df, label_encoder, num_classes = load_dataset()
    print(f"\n✓ Data loading successful!")
    print(f"  Train size: {len(train_df)}")
    print(f"  Val size: {len(val_df)}")
    print(f"  Number of classes: {num_classes}")
