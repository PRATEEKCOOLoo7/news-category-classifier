"""
Model configuration and initialization.
"""
from transformers import AutoModelForSequenceClassification, AutoConfig


def create_model(model_name='distilbert-base-uncased', num_labels=None):
    """
    Create a transformer model for sequence classification.
    
    Args:
        model_name (str): Name of the pretrained model from Hugging Face
        num_labels (int): Number of classification labels
        
    Returns:
        model: Initialized model ready for fine-tuning
    """
    if num_labels is None:
        raise ValueError("num_labels must be specified")
    
    print(f"Loading model: {model_name}")
    print(f"Number of labels: {num_labels}")
    
    # Load configuration
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(num_labels=42)
    print(f"\nModel architecture:")
    print(model)
