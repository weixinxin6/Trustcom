import torch
from torch import nn
from .layers.crf import CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding, hidden_dim, n_layers, n_class):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = embedding
        self.embedding_dim = self.embedding.embedding_dim
        self.bilstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5,
                              batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(2 * self.embedding_dim, n_class)
        self.crf = CRF(num_tags=n_class, batch_first=True)

    def forward(self, x, labels):
        embed = self.embedding(x)
        bilstm_out, _ = self.bilstm(embed)
        bilstm_out = self.dropout(bilstm_out)
        logits = self.fc(bilstm_out)
        loss = self.crf(emissions=logits, tags=labels)
        return logits, -1 * loss
