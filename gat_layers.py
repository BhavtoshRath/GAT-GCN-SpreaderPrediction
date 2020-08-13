from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter


class GraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(GraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2] # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w) # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src) # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst) # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2) # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1) # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn) # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime) # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output
