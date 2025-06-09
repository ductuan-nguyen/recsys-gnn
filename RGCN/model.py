import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, GraphConv, Linear


class RGCN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.preprocess_layers = nn.ModuleDict({
            node_type: Linear(-1, hidden_channels) for node_type in metadata[0]
        })
        self.conv1 = HeteroConv({
            edge_type: GraphConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.conv2 = HeteroConv({
            edge_type: GraphConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.post_process_layers = nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels) for node_type in metadata[0]
        })
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {node_type: self.preprocess_layers[node_type](x)
                  for node_type, x in x_dict.items()}
        x1 = self.conv1(x_dict, edge_index_dict)
        x1 = {nt: F.relu(x1[nt] + x_dict[nt]) for nt in x1}
        x1 = {nt: self.dropout_layer(x) for nt, x in x1.items()}
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {nt: F.relu(x2[nt] + x1[nt]) for nt in x2}
        x2 = {nt: self.dropout_layer(x) for nt, x in x2.items()}
        x2 = {node_type: self.post_process_layers[node_type](x)
              for node_type, x in x2.items()}
        return x2
