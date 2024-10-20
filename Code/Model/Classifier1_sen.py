import torch
import torch.nn as nn
from transformers import BertModel

class BERTEmotionClassifier_utter_sen(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, cache_dir, output_dim=3, freeze_bert=True):
        super(BERTEmotionClassifier_utter_sen, self).__init__()

        print('Loading BERT model...')
        self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        print('BERT model loaded.')
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token's representation

        x = torch.relu(self.fc(hidden_state))
        x = self.out(x)

        return x
