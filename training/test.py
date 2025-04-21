import os
import torch
import argparse
from models import MultimodalSentimentModel, MultimodalTrainer
from meld_dataset import prepare_dataloaders
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Test the sentiment model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--test_dir", type=str, default="../dataset/test", help="Path to the test directory")
    parser.add_argument("--model_path", type=str, default="../models/best_model.pth", help="Path to the saved model")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load only the test dataset
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        train_csv="../dataset/train/train_sent_emo.csv",
        train_video_dir="../dataset/train/train_splits",
        dev_csv="../dataset/dev/dev_sent_emo.csv",
        dev_video_dir="../dataset/dev/dev_splits_complete",
        test_csv="../dataset/test/test_sent_emo.csv",
        test_video_dir="../dataset/test/output_repeated_splits_test"
    )

    # Initialize model
    model = MultimodalSentimentModel().to(device)

    # Load saved weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Create trainer instance (needed for evaluation)
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,  # Not needed for testing
        val_loader=dev_loader
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_losses, test_metrics = trainer.evaluate(test_loader, phase="test")

    # Print results
    print("\nTest Results:")
    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_losses["total"]},
            {"Name": "test:emotion_loss", "Value": test_losses["emotion"]},
            {"Name": "test:sentiment_loss", "Value": test_losses["sentiment"]},
            {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
            {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
            {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
        ]
    }, indent=2))

if __name__ == "__main__":
    main()