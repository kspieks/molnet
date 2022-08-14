import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (GATv2Conv,
                                global_add_pool,
                                global_max_pool,
                                global_mean_pool,
                                )

from .layers import DMPNNConv
from .nn_utils import get_activation_function


class GNN(nn.Module):
    def __init__(self,
                 num_node_features,
                 num_edge_features,
                 gnn_depth=3,
                 gnn_hidden_size=300,
                 gnn_type='dmpnn',
                 gat_heads=1,
                 graph_pool='sum',
                 aggregation_norm=None,
                 dropout=0.0,
                 act_func='SiLU',
                 ffn_depth=3,
                 ffn_hidden_size=300,
                 num_targets=1,
                 ):
        super(GNN, self).__init__()

        self.gnn_depth = gnn_depth
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_type = gnn_type
        self.graph_pool = graph_pool
        self.aggregation_norm = aggregation_norm

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_function(act_func)

        if self.gnn_type == 'dmpnn':
            self.edge_init = nn.Linear(num_node_features + num_edge_features, self.gnn_hidden_size)
            self.edge_to_node = DMPNNConv(gnn_hidden_size=self.gnn_hidden_size)
        else:
            self.node_init = nn.Linear(num_node_features, self.gnn_hidden_size)
            self.edge_init = nn.Linear(num_edge_features, self.gnn_hidden_size)

        # gnn layers
        self.convs = torch.nn.ModuleList()
        for _ in range(self.gnn_depth):
            if self.gnn_type == 'dmpnn':
                self.convs.append(DMPNNConv(gnn_hidden_size=self.gnn_hidden_size))
            elif self.gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(in_channels=self.gnn_hidden_size,
                                            out_channels=self.gnn_hidden_size,
                                            heads=gat_heads,
                                            edge_dim=self.gnn_hidden_size)
                )
            else:
                raise ValueError(f'Undefined GNN type called {self.gnn_type}')

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn layers
        ffn = []
        if ffn_depth == 1:
            ffn.append(self.dropout)
            ffn.append(nn.Linear(self.gnn_hidden_size, num_targets))
        else:
            ffn.append(self.dropout)
            ffn.append(nn.Linear(self.gnn_hidden_size, ffn_hidden_size))

            for _ in range(ffn_depth - 2):
                ffn.extend([
                    self.activation,
                    self.dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])

            ffn.extend([
                self.activation,
                self.dropout,
                nn.Linear(ffn_hidden_size, num_targets),
            ])
        self.ffn = nn.Sequential(*ffn)

    def forward(self, data):
        # unpack data object
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # initialize node and edge features
        if self.gnn_type == 'dmpnn':
            row, col = edge_index
            edge_attr = torch.cat([x[row], edge_attr], dim=1)
            edge_attr = self.activation(self.edge_init(edge_attr))
        else:
            x = F.relu(self.node_init(x))
            edge_attr = self.activation(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # graph convolutions
        for l in range(self.gnn_depth):
            if self.gnn_type == 'gatv2':
                x_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1])
            else:
                # dmpnn passes messages along the edges
                x_h, edge_attr_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1])

            h = edge_attr_h if self.gnn_type == 'dmpnn' else x_h

            if l == self.gnn_depth - 1:
                h = self.dropout(h)
            else:
                h = self.dropout(self.activation(h))

            if self.gnn_type == 'dmpnn':
                h = h + edge_attr_h
                edge_attr_list.append(h)
            else:
                h = h + x_h
                x_list.append(h)

        # dmpnn edge -> node aggregation
        if self.gnn_type == 'dmpnn':
            h, _ = self.edge_to_node(x_list[-1], edge_index, h)

        if self.aggregation_norm:
            h = self.pool(h, batch) / self.aggregation_norm
        else:
            h = self.pool(h, batch)

        return self.ffn(h)
