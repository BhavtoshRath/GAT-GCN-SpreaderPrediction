from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from gat_layers import GraphAttention


class GAT(nn.Module):
    def __init__(self, vertex_feature, use_vertex_feature,
                 n_units=[1433, 8, 7], n_heads=[8, 1],
                 dropout=0.1, attn_dropout=0.0):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                    GraphAttention(n_heads[i], f_in=f_in,
                        f_out=n_units[i + 1], attn_dropout=attn_dropout)
                    )

    def forward(self, x, vertices, adj):
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj) # bs x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
