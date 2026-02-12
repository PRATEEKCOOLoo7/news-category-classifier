"""
Inference script for news category prediction.
"""
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json


class NewsPredictor:
    """Predictor for news category classification."""
    
    def __init__(self, model_path='models'):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to saved model directory
        """
        self.model_path = Path(model_path)
        print(f"Loading model from {self.model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load label encoder
        label_encoder_path = self.model_path / 'label_encoder.npy'
        if label_encoder_path.exists():
            self.categories = np.load(label_encoder_path, allow_pickle=True)
        else:
            # Fallback: try to infer from config
            self.categories = [f"Category_{i}" for i in range(self.model.config.num_labels)]
        
        # Set to evaluation mode
        self.model.eval()
        
        # Check device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Number of categories: {len(self.categories)}")
    
    def predict(self, text, top_k=3):
        """
        Predict the category of a news text.
        
        Args:
            text (str): News headline or headline + description
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Prediction results with category, confidence, and top_k predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()
        
        # Get top k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = {
            'text': text,
            'predicted_category': self.categories[top_indices[0]],
            'confidence': float(probs[top_indices[0]]),
            'top_predictions': [
                {
                    'category': self.categories[idx],
                    'confidence': float(probs[idx])
                }
                for idx in top_indices
            ]
        }
        
        return results
    
    def print_prediction(self, results):
        """
        Pretty print prediction results.
        
        Args:
            results (dict): Prediction results from predict()
        """
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"\nInput Text:")
        print(f"  {results['text']}")
        print(f"\nPredicted Category: {results['predicted_category']}")
        print(f"Confidence: {results['confidence']:.2%}")
        
        print(f"\nTop {len(results['top_predictions'])} Predictions:")
        for i, pred in enumerate(results['top_predictions'], 1):
            bar_length = int(pred['confidence'] * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {i}. {pred['category']:20s} {bar} {pred['confidence']:.2%}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Predict news category')
    parser.add_argument('--text', type=str, required=True,
                        help='News text to classify (headline or headline + description)')
    parser.add_argument('--model_path', type=str, default='models',
                        help='Path to saved model')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to show')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NewsPredictor(args.model_path)
    
    # Make prediction
    results = predictor.predict(args.text, top_k=args.top_k)
    
    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        predictor.print_prediction(results)


if __name__ == "__main__":
    # If no arguments, run some examples
    import sys
    if len(sys.argv) == 1:
        print("="*60)
        print("NEWS CATEGORY PREDICTOR - EXAMPLE MODE")
        print("="*60)
        print("\nUsage: python predict.py --text \"Your news headline here\"")
        print("\nRunning example predictions...\n")
        
        try:
            predictor = NewsPredictor()
            
            examples = [
                "Biden announces new climate policy initiative to reduce carbon emissions",
                "Tech giant unveils groundbreaking AI chip with unprecedented performance",
                "NBA playoffs: Lakers defeat Warriors in overtime thriller",
                "Stock market reaches all-time high amid economic recovery",
                "New study reveals health benefits of Mediterranean diet"
            ]
            
            for text in examples:
                results = predictor.predict(text, top_k=3)
                predictor.print_prediction(results)
                print()
        
        except Exception as e:
            print(f"Error: {e}")
            print("\nMake sure you have trained the model first:")
            print("  python train.py")
    else:
        main()
