import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


class DMPNNConv(MessagePassing):
    def __init__(self, gnn_hidden_size=300):
        super(DMPNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(gnn_hidden_size, gnn_hidden_size)
        self.mlp = nn.Sequential(nn.Linear(gnn_hidden_size, gnn_hidden_size),
                                 nn.BatchNorm1d(gnn_hidden_size),
                                 nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        return a_message, self.mlp(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        return F.relu(self.lin(edge_attr))


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.
    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()       # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


class DMPNNConv2(nn.Module):
    def __init__(self, gnn_hidden_size=300):
        super(DMPNNConv2, self)
        self.lin = nn.Linear(gnn_hidden_size, gnn_hidden_size)
        self.batchnorm = nn.BatchNorm1d(gnn_hidden_size)
        self.act_func = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, a2b, b2a, b2revb):
        message = edge_attr

        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        rev_message = message[b2revb]  # num_bonds x hidden
        message = a_message[b2a] - rev_message  # num_bonds x hidden

        # https://github.com/chemprop/chemprop/blob/master/chemprop/models/mpn.py#L116
        message = self.lin(message)

        return message
