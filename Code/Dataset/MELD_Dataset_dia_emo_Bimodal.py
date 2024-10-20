import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MELDDataset_dia_emo(Dataset):
    def __init__(self, text_path, aud_path, bert_model_name, split='train', cache_dir=None, max_dialogue_len=40, max_utterance_len=80, max_aud_len= 48000):
        self.text_path = text_path
        self.aud_path = aud_path
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.max_dialogue_len = max_dialogue_len
        self.max_utterance_len = max_utterance_len
        self.max_aud_len = max_aud_len
        self.data = pd.read_csv(os.path.join(text_path, f'{split}.csv'))
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
        text_ids = []
        aud = []
        attention_masks = []
        labels = []

        for i, row in dialogue_data.iterrows():
            utt_id = row['Utterance_ID']

            # get text idx
            encoded_dict = self.tokenizer.encode_plus(
                row['Utterance'],
                add_special_tokens=True,
                max_length=self.max_utterance_len,
                truncation=True,  # Explicitly enable truncation
                padding='max_length',  # Use padding='max_length'
                return_attention_mask=True,
                return_tensors='pt',
            )
            text_ids.append(encoded_dict['input_ids'])

            # get audio

            waveform, sample_rate = torchaudio.load(os.path.join(self.aud_path, f'dia{dialogue_id}_utt{utt_id}.wav'))
            aud.append(waveform)

            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(self.label2id[row['Emotion']])

        text_ids = torch.cat(text_ids, dim=0)
        aud = torch.cat(aud, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        # Padding if number of utterances is less than max_utts
        if len(text_ids) < self.max_dialogue_len:
            padding_length = self.max_dialogue_len - len(text_ids)
            padding_input_ids = torch.zeros(padding_length, self.max_utterance_len, dtype=torch.long)
            padding_aud = torch.zeros(padding_length, self.max_aud_len, dtype=torch.long)
            padding_attention_masks = torch.zeros(padding_length, self.max_utterance_len, dtype=torch.long)
            padding_labels = torch.full(size=(padding_length,), fill_value=0)

            text_ids = torch.cat([text_ids, padding_input_ids], dim=0)
            aud = torch.cat([aud, padding_aud], dim=0)
            attention_masks = torch.cat([attention_masks, padding_attention_masks], dim=0)
            labels = torch.cat([labels, padding_labels], dim=0)

        return {'text':text_ids, 'aud':aud, 'attn_mask':attention_masks, 'label':labels}
