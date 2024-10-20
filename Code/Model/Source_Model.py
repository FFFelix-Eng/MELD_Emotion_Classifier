import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sentence_length, filter_sizes, num_filters, lstm_hidden_dim,
                 output_dim, pre_trained_embeddings):
        super(TextCNN_LSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=True)
        self.conv1 = nn.Conv2d(1, num_filters, (filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, num_filters, (filter_sizes[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, num_filters, (filter_sizes[2], embedding_dim))
        self.lstm = nn.LSTM(num_filters, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def conv_block(self, x, conv_layer):
        x = conv_layer(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (x.shape[2], 1))
        return x

    def forward(self, x):
        batch_size, seq_len, sent_len = x.size()

        x = x.view(batch_size * seq_len, sent_len)
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x1 = self.conv_block(x, self.conv1)
        x2 = self.conv_block(x, self.conv2)
        x3 = self.conv_block(x, self.conv3)

        x = torch.cat((x1, x2, x3), dim=1).squeeze(2)
        x = x.view(batch_size, seq_len, -1)

        x, _ = self.lstm(x)
        x = self.dropout(x)

        x = self.fc(x)
        return x
