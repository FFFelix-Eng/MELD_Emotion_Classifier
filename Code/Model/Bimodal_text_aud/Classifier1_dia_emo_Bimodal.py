import torch
import torch.nn as nn
from transformers import BertModel

class UtteranceTextEncoder_dia_emo_Bimodal(nn.Module):
    def __init__(self, bert_model_name, cache_dir=None, freeze_bert=True):
        super(UtteranceTextEncoder_dia_emo_Bimodal, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
                                nn.ReLU(),)

    def forward(self, input_ids, attention_mask):
        # Check if any attention_mask is all zeros along the last dimension
        all_zero_mask = (attention_mask.sum(dim=1) == 0)

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # If attention_mask is all zeros, set corresponding outputs to zero
        bert_outputs.last_hidden_state[all_zero_mask, :, :] = 0

        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token's representation

        output = self.fc(cls_output)

        return output

class AudEncoder(nn.Module):
    def __init__(self, aud_length=48000, target_emb_len=768):
        super(AudEncoder, self).__init__()
        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, stride=8, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(4),
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.Flatten(),
            nn.Linear(16 * (300), target_emb_len)  # 使音频嵌入与文本嵌入具有相同维度
        )

    def forward(self, x):
        # Conv need the channel dim
        return self.audio_feature_extractor(x.unsqueeze(1))

class DialogueClassifier_Bimodal(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim=7, rnn_type='LSTM', cache_dir=None, freeze_bert=True):
        super(DialogueClassifier_Bimodal, self).__init__()
        self.text_encoder = UtteranceTextEncoder_dia_emo_Bimodal(bert_model_name, cache_dir=cache_dir, freeze_bert=freeze_bert)
        self.aud_encoder = AudEncoder(target_emb_len=self.text_encoder.bert.config.hidden_size)

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.text_encoder.bert.config.hidden_size * 2, hidden_dim, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unsupported RNN type: choose from 'LSTM'")

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=2)  # Apply softmax along the last dimension

    def forward(self, text_ids, aud, attention_mask):
        batch_size, max_utts, max_len = text_ids.size()
        text_ids = text_ids.view(batch_size * max_utts, max_len)
        aud = aud.view(batch_size * max_utts, -1)
        attention_mask = attention_mask.view(batch_size * max_utts, max_len)

        utterance_encodings = self.text_encoder(text_ids, attention_mask)
        utterance_encodings = utterance_encodings.view(batch_size, max_utts, -1)

        aud_encodings = self.aud_encoder(aud)
        aud_encodings = aud_encodings.view(batch_size, max_utts, -1)

        combined_embeddings = torch.cat((utterance_encodings, aud_encodings), dim=2)

        rnn_outputs, _ = self.rnn(combined_embeddings)

        # Reshape before passing through the fully connected layer
        rnn_outputs = rnn_outputs.contiguous().view(batch_size * max_utts, -1)
        output = self.fc(rnn_outputs)

        # Reshape back to [batch_size, max_utts, output_dim]
        output = output.view(batch_size, max_utts, -1)
        # output = self.softmax(output)  # Apply softmax

        return output