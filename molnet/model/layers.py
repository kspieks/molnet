import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class DMPNNConv(MessagePassing):
    def __init__(self, gnn_hidden_size=300):
        super(DMPNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(gnn_hidden_size, gnn_hidden_size)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        return a_message, self.lin(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        return edge_attr
