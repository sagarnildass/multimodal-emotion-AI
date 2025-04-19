from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2

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
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        try:
            print(f"Loading video from {video_path}")
        except Exception as e:
            raise ValueError(f"Video Error: {str(e)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4"""
        path = os.path.join(self.video_dir, video_filename)
        video_path = os.path.exists(path)

        if not video_path:
            raise FileNotFoundError(f"Video file {video_filename} does not exist")
                
        text_inputs = self.tokenizer(row["Utterance"], 
                                     return_tensors="pt",
                                     padding='max_length',
                                     truncation=True,
                                     max_length=128)
        
        video_frames = self._load_video_frames(path)
        
        print(text_inputs)
        
        


if __name__ == "__main__":
    meld = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    print(meld[0])
    
