import torch.nn as nn
from .layers.crf import CRF
from transformers import BertModel, BertPreTrainedModel

class BertGRUCrfForNer(BertPreTrainedModel):
    def __init__(self, config, lstm_hidden_dim, n_layers):
        super(BertGRUCrfForNer, self).__init__(config, lstm_hidden_dim, n_layers)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.bilstm = nn.LSTM(config.hidden_size, lstm_hidden_dim, num_layers=n_layers, bidirectional=True,
        #                       batch_first=True, dropout=0.5)
        self.gru = nn.GRU(config.hidden_size, lstm_hidden_dim, num_layers=n_layers, bidirectional=True,
                               batch_first=True, dropout=0.5)

        self.classifier = nn.Linear(2 * lstm_hidden_dim, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        #bilstm_outputs, _ = self.bilstm(sequence_output)
        bilstm_outputs, _ = self.gru(sequence_output)
        
        bilstm_outputs = self.dropout(bilstm_outputs)
        logits = self.classifier(bilstm_outputs)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores