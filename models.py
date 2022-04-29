import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(2*nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        print(sum(adj[0][0]))
        outputs1 = F.relu(self.gc1(x, adj[0]))
        outputs2 = F.relu(self.gc1(x, torch.eye(adj[0].shape[0])))
        outputs3 = torch.cat((outputs1,outputs2),1)

        outputs4 = F.dropout(outputs3, self.dropout, training=self.training)
        outputs5 = self.gc2(outputs4, adj[1])
        return F.log_softmax(outputs5, dim=1)
        #return self.out_softmax(outputs5)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)
