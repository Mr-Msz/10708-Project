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

    def forward(self, x, target_feats, adj):
        outputs1 = F.relu(self.gc1(x, adj[0]))
        outputs2 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs3 = self.gc2(outputs2, adj[1])
        return F.log_softmax(outputs3, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)

class HybridMethod(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.linear1 = nn.Linear(nfeat + nhid, nfeat + nhid)
        self.linear2 = nn.Linear(nfeat + nhid, nfeat + nhid)
        self.linear3 = nn.Linear(nfeat + nhid, nfeat)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, target_feats, adj):
        outputs1 = F.relu(self.gc1(x, adj[0]))
        outputs2 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs3 = self.gc2(outputs2, adj[1])
        outputs4 = F.relu(self.linear1(torch.cat([outputs3, target_feats], axis = 1)))
        outputs5 = F.relu(self.linear2(outputs4))
        outputs6 = self.linear3(outputs5)
        return F.log_softmax(outputs6, dim=1) 

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)
