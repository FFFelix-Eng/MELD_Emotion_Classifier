import torch
import torch.nn as nn
from transformers import BertModel

class BERTUtteranceEncoder_dia_emo(nn.Module):
    def __init__(self, bert_model_name, cache_dir=None, freeze_bert=True):
        super(BERTUtteranceEncoder_dia_emo, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

            # Unfreeze pooler layer parameters
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

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

class DialogueClassifier(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim=7, rnn_type='LSTM', cache_dir=None, freeze_bert=True):
        super(DialogueClassifier, self).__init__()
        self.utterance_encoder = BERTUtteranceEncoder_dia_emo(bert_model_name, cache_dir=cache_dir, freeze_bert=freeze_bert)

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.utterance_encoder.bert.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unsupported RNN type: choose from 'LSTM'")

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=2)  # Apply softmax along the last dimension

    def forward(self, input_ids, attention_mask):
        batch_size, max_utts, max_len = input_ids.size()
        input_ids = input_ids.view(batch_size * max_utts, max_len)

        attention_mask = attention_mask.view(batch_size * max_utts, max_len)

        utterance_encodings = self.utterance_encoder(input_ids, attention_mask)
        utterance_encodings = utterance_encodings.view(batch_size, max_utts, -1)


        rnn_outputs, _ = self.rnn(utterance_encodings)

        # Reshape before passing through the fully connected layer
        rnn_outputs = rnn_outputs.contiguous().view(batch_size * max_utts, -1)
        output = self.fc(rnn_outputs)

        # Reshape back to [batch_size, max_utts, output_dim]
        output = output.view(batch_size, max_utts, -1)
        # output = self.softmax(output)  # Apply softmax

        return output