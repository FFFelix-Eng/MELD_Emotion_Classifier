import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer

class MELDDataset(Dataset):
    def __init__(self, data_path, bert_model_name, cache_dir, split='train', max_len=512):
        self.data_path = data_path
        self.split = split
        self.max_len = max_len

        # Load preprocessed data
        self.data = pd.read_csv(os.path.join(data_path, f'{split}.csv'))
        self.utterances = self.data['Utterance'].tolist()
        self.labels = self.data['Emotion'].tolist()

        # Initialize BERT tokenizer
        print('Loading BertTokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, cache_dir=cache_dir)
        print('Loaded.')

        # Predefined emotion label dictionary
        self.label_dict = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
        self.num_classes = len(self.label_dict)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        label = self.labels[idx]

        # Tokenize using BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            utterance,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Convert label to integer
        label_idx = self.label_dict[label]

        return input_ids, attention_mask, torch.tensor(label_idx)

