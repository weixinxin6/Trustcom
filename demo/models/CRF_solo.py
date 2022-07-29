import torch
from torch import nn
from .layers.crf import CRF
class CRF_solo(nn.Module):
    def __init__(self, embedding, n_class):
        super(CRF_solo, self).__init__()
        self.embedding = embedding
        self.embedding_dim = self.embedding.embedding_dim

        self.fc = nn.Linear(self.embedding_dim, n_class)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(self.embedding_dim, n_class)
        self.crf = CRF(num_tags=n_class, batch_first=True)

    def forward(self, x, labels):
        embed = self.embedding(x)
        logits = self.fc(embed)
        loss = self.crf(emissions=logits, tags=labels)
        return logits, -1 * loss
