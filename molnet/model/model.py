import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATv2Conv,
                                global_add_pool,
                                global_max_pool,
                                global_mean_pool,
                                )

from .layers import DMPNNConv
from .nn_utils import get_activation_function


class GNN(nn.Module):
    def __init__(self, args, num_node_features, num_edge_features):
        super(GNN, self).__init__()

        self.args = args
        self.gnn_depth = args.gnn_depth
        self.gnn_hidden_size = args.gnn_hidden_size
        self.gnn_type = args.gnn_type
        self.graph_pool = args.graph_pool

        self.dropout = nn.Dropout(args.dropout)
        self.activation = get_activation_function(args.act_func)

        if self.gnn_type == 'dmpnn':
            self.edge_init = nn.Linear(num_node_features + num_edge_features, self.gnn_hidden_size)
            self.edge_to_node = DMPNNConv(args)
        else:
            self.node_init = nn.Linear(num_node_features, self.gnn_hidden_size)
            self.edge_init = nn.Linear(num_edge_features, self.gnn_hidden_size)

        # gnn layers
        self.convs = torch.nn.ModuleList()
        for _ in range(self.gnn_depth):
            if self.gnn_type == 'dmpnn':
                self.convs.append(DMPNNConv(args))
            elif self.gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(in_channels=self.gnn_hidden_size,
                                            out_channels=self.gnn_hidden_size,
                                            heads=args.gat_heads, 
                                            edge_dim=self.gnn_hidden_size)
                )
            else:
                raise ValueError(f'Undefined GNN type called {self.gnn_type}')

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif args.graph_pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn layers
        ffn = []
        if args.ffn_depth == 1:
            ffn.append(self.dropout)
            ffn.append(nn.Linear(args.gnn_hidden_size, len(args.targets)))
        else:
            ffn.append(self.dropout)
            ffn.append(nn.Linear(args.gnn_hidden_size, args.ffn_hidden_size))

            for _ in range(args.ffn_depth - 2):
                ffn.extend([
                    self.activation,
                    self.dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])

            ffn.extend([
                self.activation,
                self.dropout,
                nn.Linear(args.ffn_hidden_size, len(args.targets)),
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

        if self.args.aggregation_norm:
            h = self.pool(h, batch) / self.args.aggregation_norm
        else:
            h = self.pool(h, batch)

        return self.ffn(h)
