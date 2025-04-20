import torch.nn as nn
import torch
from transformers import BertModel
from torchvision import models as vision_models

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



        
        