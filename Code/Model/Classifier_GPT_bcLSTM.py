from torch import nn
from Code.Model.GPT_embedding_generator import GPTEmbeddingGenerator


class DialogueClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim=7, rnn_type='LSTM'):
        super(DialogueClassifier, self).__init__()

        self.embedding_generator = GPTEmbeddingGenerator()

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.embedding_generator.utterance_embedding_len, hidden_dim, batch_first=True,
                               bidirectional=True)
        else:
            raise ValueError("Unsupported RNN type: choose from 'LSTM'")

        self.LayerNorm = nn.LayerNorm([self.embedding_generator.utterance_embedding_len])

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=2)  # Apply softmax along the last dimension



    def forward(self, x, device):
        # batch_size = x.
        # max_dialogue_len= input_ids.size()
        # input_ids = input_ids.view(batch_size * max_utts, max_len)
        #
        # attention_mask = attention_mask.view(batch_size * max_utts, max_len)

        # utterance_encodings = self.utterance_encoder(input_ids, attention_mask)
        # utterance_encodings = utterance_encodings.view(batch_size, max_utts, -1)

        embeds, batch_size, max_dialogue_len = self.embedding_generator.get_batch_embeddings(x)

        rnn_outputs, _ = self.rnn(self.LayerNorm(embeds.to(device)))

        # Reshape before passing through the fully connected layer
        rnn_outputs = rnn_outputs.contiguous().view(batch_size * max_dialogue_len, -1)
        output = self.fc(rnn_outputs)

        # Reshape back to [batch_size, max_utts, output_dim]
        output = output.view(batch_size, max_dialogue_len, -1)
        # output = self.softmax(output)  # Apply softmax

        return output
