"""
Preprocessing and tokenization module.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NewsDataset(Dataset):
    """PyTorch Dataset for news classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of text strings
            labels (list): List of label integers
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_df, val_df, model_name='distilbert-base-uncased', 
                        batch_size=16, max_length=128):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data
        model_name (str): Name of the pretrained model
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = NewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = NewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ DataLoaders created")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test tokenization
    from data_loader import load_dataset
    
    train_df, val_df, _, _ = load_dataset()
    train_loader, val_loader, tokenizer = create_data_loaders(train_df, val_df)
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shape test:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
