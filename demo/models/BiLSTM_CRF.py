import torch
from torch import nn
from .layers.crf import CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_class,embedding_):
        super(BiLSTM_CRF, self).__init__()
        #print("hidden_dim:")
        #print(hidden_dim)#100

        #print("n_layers:")
        #print(n_layers)#1

        #print("n_class:")
        #print(n_class)#29

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #self.embedding = embedding

        #num_embeddings=vocab_size, embedding_dim=embedding_dim
        self.embedding = embedding_
        self.embedding_dim = self.embedding.embedding_dim
        

        '''该参数是影响输⼊数据的格式问题，
           若batch_first = True, 则输⼊数据的格式应为：（Batch_size, Length, input_dim），
           否则为(Length, Batch_size, input_dim)'''
        self.bilstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5,
                              batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(2 * self.embedding_dim, n_class)
        self.crf = CRF(num_tags=n_class, batch_first=True)

    def forward(self, x, labels):
        #print(x)#单词对应的索引
        #print(labels)#标签对应的索引

        #print("输入维度：")
        #print(self.embedding_dim)#100
        
        embed = self.embedding(x)
        #print("embed.size:")
        #print(embed.size())#[16, X, 100]
        #print(embed)

        #输⼊数据的格式应为：（Batch_size, Length, input_dim）
        #print(x.ndim)#维度：2
        
        #print("x.size():")
        #print(x.size())#[16, X]
        #print(len(x))#16
        
        #x = x.view(len(x),1,-1)#592=16×37
        #x = x.float()

        #print(labels.size())#[16, 37]
        #h_0格式(num_directions * num_layers, batch_size, hidden_size)
        #c_0格式(num_directions * num_layers, batch_size, hidden_size)
        #num_directions双向LSTM为2，单向为1

        ###batch_size, seq_len = x.size()[0], x.size()[1]
        #print(batch_size)#16
        #print(seq_len)#37
        
        ###h_0 = torch.randn(2 * self.n_layers, batch_size, self.hidden_dim)
        ###c_0 = torch.randn(2 * self.n_layers, batch_size, self.hidden_dim)
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)


        ###bilstm_out, _ = self.bilstm(x, (h_0, c_0))  # output(5, 30, 64)
        #pred = self.linear(output) # (5, 30, 1)
        #pred = pred[:, -1, :]  # (5, 1)
        
        #input (16,37,100)
        #x = x.unsqueeze(dim=2)
        #print(x.ndim)
        #print(x.size())

        bilstm_out, _ = self.bilstm(embed)
        bilstm_out = self.dropout(bilstm_out)
        logits = self.fc(bilstm_out)
        loss = self.crf(emissions=logits, tags=labels)
        return logits, -1 * loss
    
