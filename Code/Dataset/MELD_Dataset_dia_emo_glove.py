import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class MELDDataset_dia_emo_glove(Dataset):
    def __init__(self, data_path, glove_file_path, split='train', max_utterance_len=50, max_utts=70, cache_dir=None):
        self.data_path = data_path
        self.split = split
        self.max_utterance_len = max_utterance_len
        self.max_dialogue_len = max_utts
        self.data = pd.read_csv(os.path.join(data_path, f'{split}.csv'))
        self.num_classes = self.data['Emotion'].nunique()
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
        self.dialogue_ids = self.data['Dialogue_ID'].unique()

        self.vocab, self.embedding_matrix = self.load_glove_embeddings(glove_file_path)

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, idx):
        dialogue_id = self.dialogue_ids[idx]
        dialogue_data = self.data[self.data['Dialogue_ID'] == dialogue_id]
        input_ids = []
        attention_masks = []
        labels = []

        for i, row in dialogue_data.iterrows():
            tokens = row['Utterance'].split()
            token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            token_ids = token_ids[:self.max_utterance_len] + [self.vocab['<PAD>']] * (self.max_utterance_len - len(token_ids))
            attention_mask = [1 if token != self.vocab['<PAD>'] else 0 for token in token_ids]
            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
            labels.append(self.label2id[row['Emotion']])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Padding if number of utterances is less than max_utts
        if len(input_ids) < self.max_dialogue_len:
            padding_length = self.max_dialogue_len - len(input_ids)
            padding_input_ids = torch.zeros(padding_length, self.max_utterance_len, dtype=torch.long)
            padding_attention_masks = torch.zeros(padding_length, self.max_utterance_len, dtype=torch.long)
            padding_labels = torch.zeros(padding_length, dtype=torch.long)

            input_ids = torch.cat([input_ids, padding_input_ids], dim=0)
            attention_masks = torch.cat([attention_masks, padding_attention_masks], dim=0)
            labels = torch.cat([labels, padding_labels], dim=0)

        return input_ids, attention_masks, labels

    def load_glove_embeddings(self, glove_file_path):
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[word] = vector

        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word in embeddings_index.keys():
            vocab[word] = len(vocab)

        embedding_dim = len(next(iter(embeddings_index.values())))
        embedding_matrix = np.zeros((len(vocab), embedding_dim))
        for word, idx in vocab.items():
            vector = embeddings_index.get(word)
            if vector is not None:
                embedding_matrix[idx] = vector
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

        return vocab, torch.tensor(embedding_matrix, dtype=torch.float32)
