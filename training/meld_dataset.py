from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class MELDDataset(Dataset):
    def __init__(self, csv_path: str, video_dir: str):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # anger, disgust, sadness, joy, neutral, surprise and fear
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        self.sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

    def __len__(self):
        return len(self.data)

    
        

if __name__ == "__main__":
    dataset = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    
