import os 
import argparse
import torch
import torchaudio
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sentiment model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    parser.add_argument("--train_dir", type=str, default="../dataset/train", help="Path to the training directory")
    parser.add_argument("--val_dir", type=str, default="../dataset/dev", help="Path to the development directory")
    parser.add_argument("--test_dir", type=str, default="../dataset/test", help="Path to the test directory")

    parser.add_argument("--model_dir", type=str, default="../models", help="Path to save the model")

    return parser.parse_args()

def main():
    # Install FFMPEG if not installed

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        memory_used = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_video_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_video_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_video_dir=os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size
    )

    print(f"Train CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Train Video Directory: {os.path.join(args.train_dir, 'train_splits')}")

    model = MultimodalSentimentModel().to(device)

    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

    best_val_loss = float('inf')

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "test_losses": [],
        "epochs": [],
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_losses = trainer.train_epoch()
        val_losses, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_losses["total"])
        metrics_data["val_losses"].append(val_losses["total"])
        metrics_data["epochs"].append(epoch)

        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_losses["total"]},
                {"Name": "validation:loss", "Value": val_losses["total"]},
                {"Name": "validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            # Create model directory if it doesn't exist
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")

    test_losses, test_metrics = trainer.evaluate(test_loader, phase="test")

    metrics_data["test_losses"].append(test_losses["total"])

    print(json.dumps({
            "metrics": [
                {"Name": "test:loss", "Value": test_losses["total"]},
                {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
                {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
                {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
                {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
            ]
        }))

    
if __name__ == "__main__":
    main()
    