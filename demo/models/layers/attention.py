import math

import torch
import torch.nn as nn
nn.MultiheadAttention
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.cpu().data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

        # attn_output, attention = self.attention_net(output, final_hidden_state)

    def attention_net(self, x, query, mask=None):  # 软性注意力机制（key=value=x）
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, lstm_output, final_state):
        return self.attention_net(lstm_output, final_state)