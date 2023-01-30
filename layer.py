import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rewritten_GCN import GCNConv
from torch_geometric.nn import GINConv,GATConv


def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

class TorGNN(nn.Module):
    def __init__(self, aggregator, feature, hidden1, hidden2, decoder1, dropout):
        super(TorGNN, self).__init__()

        
        if aggregator == 'GCN':
            self.encoder_o1 = GCNConv(feature, hidden2)
            #self.encoder_o2 = GCNConv(hidden1, hidden2)
            #self.encoder_o3 = GCNConv(hidden1, hidden2)

        self.decoder1 = nn.Linear(hidden2 * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)
        
        self.lin1 = nn.Linear(10, 1)
        self.lin2 = nn.Linear(10, 1)
        self.lin3 = nn.Linear(10, 1)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, data_o, idx):

        x_o, adj,curvature = data_o.x, data_o.edge_index,data_o.curva
        curva1 = torch.abs(curvature).float()
        curva1 = F.dropout(curva1, self.dropout, training=self.training)
        
        x1_o = F.relu(self.encoder_o1(x_o, adj, curva1))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        

        entity1 = x1_o[idx[0]]
        entity2 = x1_o[idx[1]]
        
        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)
        feature = torch.cat((add, product, concatenate), dim=1)

        # decoder
        log = F.relu(self.decoder1(feature))
        log = self.decoder2(log)

        return log
