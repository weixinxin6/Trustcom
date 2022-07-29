import torch.nn as nn
from .layers.crf import CRF
from .layers.attention import Attention
from transformers import BertModel, BertPreTrainedModel

class BertBiLSTMAttenCrfForNer(BertPreTrainedModel):
    def __init__(self, config, lstm_hidden_dim, n_layers):
        super(BertBiLSTMAttenCrfForNer, self).__init__(config, lstm_hidden_dim, n_layers)
        #print("lstm_hidden_dim:")
        #print(lstm_hidden_dim)#100
        #print("n_layers:")
        #print(n_layers)#1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(config.hidden_size, lstm_hidden_dim, num_layers=n_layers, bidirectional=True,
                              batch_first=True, dropout=0.5)
        self.classifier = nn.Linear(2 * lstm_hidden_dim, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.attention = nn.MultiheadAttention(2 * lstm_hidden_dim, 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        bilstm_outputs, (final_hidden_state, final_cell_state) = self.bilstm(sequence_output)
        bilstm_outputs = self.dropout(bilstm_outputs)
        attn_output, _ = self.attention(bilstm_outputs, bilstm_outputs, bilstm_outputs)
        logits = self.classifier(attn_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores