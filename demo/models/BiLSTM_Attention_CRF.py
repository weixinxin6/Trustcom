import torch
from torch import nn
from .layers.crf import CRF
from .layers.attention import Attention
class BiLSTM_Atten_CRF(nn.Module):
    def __init__(self, embedding, hidden_dim, n_layers, n_class):
        super(BiLSTM_Atten_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = embedding
        self.embedding_dim = self.embedding.embedding_dim
        self.bilstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5,
                              batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.dropout = nn.Dropout(0.5)
        self.attention = nn.MultiheadAttention(2 * hidden_dim, 2)
        self.classifier = nn.Linear(2 * self.embedding_dim, n_class)
        self.crf = CRF(num_tags=n_class, batch_first=True)

    def forward(self, x, labels):
        embed = self.embedding(x)
        bilstm_out, (final_hidden_state, final_cell_state) = self.bilstm(embed)
        bilstm_out = self.dropout(bilstm_out)
        attn_output, _ = self.attention(bilstm_out, bilstm_out, bilstm_out)
        logits = self.fc(attn_output)
        loss = self.crf(emissions=logits, tags=labels)
        return logits, -1 * loss
