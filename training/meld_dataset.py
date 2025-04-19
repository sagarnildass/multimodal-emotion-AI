from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch

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
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")
            
            #  try and read the first frame to validate the video
            ret, frame = cap.read()

            if not ret or frame is None:
                raise ValueError(f"Video is empty or invalid: {video_path}")

            # Reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break
                
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
                
        except Exception as e:
            raise ValueError(f"Video Error: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames could be extracted from video: {video_path}")

        # pad or truncate to 30 frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames).permute(0, 3, 1, 2))

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
    
