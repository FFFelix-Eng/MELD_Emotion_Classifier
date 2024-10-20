import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MELDDataset_dia_emo(Dataset):
    def __init__(self, data_path, bert_model_name, split='train', cache_dir=None, max_dialogue_len=40, max_utterance_len=80):
        self.data_path = data_path
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.max_dialogue_len = max_dialogue_len
        self.max_utterance_len = max_utterance_len
        self.data = pd.read_csv(os.path.join(data_path, f'{split}.csv'))
        self.label2id = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
        self.data = self.data.groupby('Dialogue_ID').apply(lambda x: x[:self.max_dialogue_len])
        self.data.reset_index(drop=True, inplace=True)
        self.dialogue_list = self.data['Dialogue_ID'].unique().tolist()


    def __len__(self):
        return len(self.dialogue_list)

    def __getitem__(self, idx):
        dialogue_id = self.dialogue_list[idx]
        dialogue_data = self.data[self.data['Dialogue_ID'] == dialogue_id]
        input_ids = []
        attention_masks = []
        labels = []

        for i, row in dialogue_data.iterrows():
            encoded_dict = self.tokenizer.encode_plus(
                row['Utterance'],
                add_special_tokens=True,
                max_length=self.max_utterance_len,
                truncation=True,  # Explicitly enable truncation
                padding='max_length',  # Use padding='max_length'
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(self.label2id[row['Emotion']])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        # Padding if number of utterances is less than max_utts
        if len(input_ids) < self.max_dialogue_len:
            padding_length = self.max_dialogue_len - len(input_ids)
            padding_input_ids = torch.zeros(padding_length, self.max_utterance_len, dtype=torch.long)
            padding_attention_masks = torch.zeros(padding_length, self.max_utterance_len, dtype=torch.long)
            padding_labels = torch.full(size=(padding_length,), fill_value=0)

            input_ids = torch.cat([input_ids, padding_input_ids], dim=0)
            attention_masks = torch.cat([attention_masks, padding_attention_masks], dim=0)
            labels = torch.cat([labels, padding_labels], dim=0)

        return input_ids, attention_masks, labels
