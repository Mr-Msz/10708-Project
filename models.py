import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        outputs1 = F.relu(self.gc1(x, adj[0]))
        outputs2 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs3 = self.gc2(outputs2, adj[1])
        return F.log_softmax(outputs3, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)
