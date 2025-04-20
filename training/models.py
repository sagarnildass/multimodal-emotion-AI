import torch.nn as nn
import torch
from transformers import BertModel
from torchvision import models as vision_models
from meld_dataset import MELDDataset
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # We will not train the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Project to 128 dimensions (BERT output is 768)
        self.projection = nn.Linear(self.bert.config.hidden_size, 128)
    
    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids, attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Project to 128 dimensions
        return self.projection(pooled_output)
    
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # We will not train the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        
        # Only the linear layer is trainable
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # [Batch_size, Frames, Channels, Height, Width] --> [Batch_size, Channels, Frames, Height, Width]
        x = x.transpose(1, 2)
        return self.backbone(x)
    

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower Level Features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Higher Level Features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Freeze the conv layers
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
        # Only the linear layer is trainable
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # [batch_size, 1, 64, 300] --> [batch_size, 64, 300]
        x = x.squeeze(1)
        features = self.conv_layers(x) 

        # [batch_size, 128, 1] --> [batch_size, 128]
        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7) # 7 emotions - anger, disgust, fear, joy, neutral, sadness, surprise
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # 3 sentiments - negative, neutral, positive
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features
        combined_features = torch.cat([
            text_features, 
            video_features, 
            audio_features], 
            dim=1) # [batch_size, 128 * 3]
        
        # Fusion Layer
        fused_features = self.fusion_layer(combined_features)

        # Emotion Classification
        emotion_output = self.emotion_classifier(fused_features)

        # Sentiment Classification
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            "emotion_output": emotion_output,
            "sentiment_output": sentiment_output
        }
    
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log Dataset sizes
        print(f"Training Dataset Size: {len(train_loader.dataset):,}")
        print(f"Validation Dataset Size: {len(val_loader.dataset):,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        base_dir = "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        #  Optimizer
        self.optimizer = torch.optim.Adam([
            {"params": model.text_encoder.parameters(), "lr": 8e-6},
            {"params": model.video_encoder.parameters(), "lr": 8e-5},
            {"params": model.audio_encoder.parameters(), "lr": 8e-5},
            {"params": model.fusion_layer.parameters(), "lr": 5e-4},
            {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
            {"params": model.sentiment_classifier.parameters(), "lr": 5e-4}
        ], weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=2,
            verbose=True
        )

        self.current_train_losses = None

        # Loss Function
        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else: # Validation phase
            self.writer.add_scalar("loss/total/train", self.current_train_losses["total"], self.global_step)
            self.writer.add_scalar("loss/total/val", losses["total"], self.global_step)
            self.writer.add_scalar("loss/emotion/train", self.current_train_losses["emotion"], self.global_step)
            self.writer.add_scalar("loss/emotion/val", losses["emotion"], self.global_step)
            self.writer.add_scalar("loss/sentiment/train", self.current_train_losses["sentiment"], self.global_step)
            self.writer.add_scalar("loss/sentiment/val", losses["sentiment"], self.global_step)
        
        if metrics:
            self.writer.add_scalar(f"{phase}/emotion_precision", metrics["emotion_precision"], self.global_step)
            self.writer.add_scalar(f"{phase}/emotion_accuracy", metrics["emotion_accuracy"], self.global_step)
            self.writer.add_scalar(f"{phase}/sentiment_precision", metrics["sentiment_precision"], self.global_step)
            self.writer.add_scalar(f"{phase}/sentiment_accuracy", metrics["sentiment_accuracy"], self.global_step)
            

    def train_epoch(self):
        self.model.train()
        running_loss = {"total": 0.0, "emotion": 0.0, "sentiment": 0.0}
        
        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)

            emotion_labels = batch["emotion_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            
            # Zero Gradients
            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Compute Loss using raw logits
            emotion_loss = self.emotion_criterion(outputs["emotion_output"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs["sentiment_output"], sentiment_labels)

            # Backward Pass
            total_loss = emotion_loss + sentiment_loss
            total_loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update Weights
            self.optimizer.step()

            # Track Losses
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()

            self.log_metrics({
                "total": total_loss.item(),
                "emotion": emotion_loss.item(),
                "sentiment": sentiment_loss.item()
            })

            self.global_step += 1
            
        # Calculate Averages
        return {k: v / len(self.train_loader) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {"total": 0.0, "emotion": 0.0, "sentiment": 0.0}

        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device

                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
                }
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)

                emotion_labels = batch["emotion_label"].to(device)
                sentiment_labels = batch["sentiment_label"].to(device)

                # Forward Pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Compute Loss
                emotion_loss = self.emotion_criterion(outputs["emotion_output"], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs["sentiment_output"], sentiment_labels)

                all_emotion_preds.extend(outputs["emotion_output"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())

                all_sentiment_preds.extend(outputs["sentiment_output"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track Losses
                losses["total"] += (emotion_loss + sentiment_loss).item()
                losses["emotion"] += emotion_loss.item()
                losses["sentiment"] += sentiment_loss.item()
              
        # Calculate Averages
        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}

        # Compute the precision and accuracy
        emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average="weighted")
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)

        sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average="weighted")
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)

        self.log_metrics(avg_loss, {
            "emotion_precision": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy
        }, phase=phase)

        if phase == "val":
            self.scheduler.step(avg_loss["total"])

        return avg_loss, {
            "emotion_precision": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy
        }
                

if __name__ == "__main__":
    # Training Data only
    dataset = MELDDataset(
        csv_path="../dataset/train/train_sent_emo.csv",
        video_dir="../dataset/train/train_splits"
    )

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        "input_ids": sample["text_inputs"]["input_ids"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0)
    }

    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        
        emotion_probs = torch.softmax(outputs["emotion_output"], dim=1)[0]
        sentiment_probs = torch.softmax(outputs["sentiment_output"], dim=1)[0]

    emotion_map = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6
    }

    emotion_map = {v: k for k, v in emotion_map.items()}

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    print("Emotion:")
    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.4f}")

    print("\nSentiment:")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.4f}")



        
        
        
        
        
        
        