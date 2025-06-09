import torch
from torch import nn
from torch_geometric.loader import NeighborLoader


def load_data(path):
    """Load and preprocess heterogeneous graph data from a file."""
    data = torch.load(path)
    # Process 'item' node features
    num_price_buckets = int(data['item'].price.max().item()) + 1
    price_embedding_dim = 8
    price_embedding = nn.Embedding(num_price_buckets, price_embedding_dim)
    price_emb = price_embedding(data['item'].price)
    name_max = torch.max(torch.abs(data['item'].name))
    if name_max > 0:
        data['item'].name = data['item'].name / name_max
    data['item'].x = torch.cat([data['item'].name, price_emb], dim=1).detach().requires_grad_(False)

    # Process 'user' node features
    time_window_max = torch.max(torch.abs(data['user'].time_window_feats))
    if time_window_max > 0:
        data['user'].time_window_feats = data['user'].time_window_feats / time_window_max
    data['user'].x = data['user'].time_window_feats

    # Initialize 'url' and 'category' node embeddings
    embedding_dim = 16
    data['url'].x = torch.randn(data['url'].num_nodes, embedding_dim)
    data['category'].x = torch.randn(data['category'].num_nodes, embedding_dim)
    return data


def create_train_loader(data, batch_size, num_workers):
    """Create a NeighborLoader for training."""
    return NeighborLoader(
        data,
        num_neighbors={key: [3, 2, 2] for key in data.edge_types},
        input_nodes=('user', torch.arange(data['user'].num_nodes)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def create_inference_loader(data, batch_size, num_workers):
    """Create a NeighborLoader for inference (user embedding extraction)."""
    return NeighborLoader(
        data,
        num_neighbors={key: [15, 10, 5] for key in data.edge_types},
        input_nodes=('user', torch.arange(data['user'].num_nodes)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
