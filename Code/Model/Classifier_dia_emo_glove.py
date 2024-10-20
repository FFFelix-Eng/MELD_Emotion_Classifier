import torch.nn as nn
import torch.nn.functional as F
import torch

class TextCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, num_filters, filter_sizes, lstm_hidden_dim, output_dim):
        super(TextCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = False # 如果不希望微调嵌入层，可以保持为False

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        self.lstm = nn.LSTM(num_filters * len(filter_sizes), lstm_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def conv_and_pool(self, x, conv, batch_norm):
        x = F.relu(batch_norm(conv(x))).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text):
        batch_size, max_utts, max_len = text.size()
        text = text.view(batch_size * max_utts, max_len)

        embedded = self.embedding(text).unsqueeze(1)  # shape: (batch_size*max_utts, 1, max_len, embedding_dim)
        conved = [self.conv_and_pool(embedded, conv, batch_norm) for conv, batch_norm in zip(self.convs, self.batch_norms)]
        cat = torch.cat(conved, dim=1)  # shape: (batch_size*max_utts, num_filters*len(filter_sizes))
        cat = cat.view(batch_size, max_utts, -1)  # shape: (batch_size, max_utts, num_filters*len(filter_sizes))

        lstm_out, _ = self.lstm(cat)  # shape: (batch_size, max_utts, lstm_hidden_dim*2)
        output = self.fc(lstm_out)  # shape: (batch_size, max_utts, output_dim)
        return output
