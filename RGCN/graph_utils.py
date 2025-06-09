import os
import time
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from utils import convert_list_with_bracket_without_comma_string_to_list


def load_dataframes(data_dir):
    """Load raw parquet files and preprocess list-like columns."""
    buy   = pd.read_parquet(os.path.join(data_dir, "product_buy.parquet"))
    add   = pd.read_parquet(os.path.join(data_dir, "add_to_cart.parquet"))
    remove= pd.read_parquet(os.path.join(data_dir, "remove_from_cart.parquet"))
    visit = pd.read_parquet(os.path.join(data_dir, "page_visit.parquet"))
    search= pd.read_parquet(os.path.join(data_dir, "search_query.parquet"))
    prod  = pd.read_parquet(os.path.join(data_dir, "product_properties.parquet"))
    
    # normalize list-like columns
    for df, col in [(search, "query"), (prod, "name")]:
        df[col] = df[col].apply(
            lambda x: convert_list_with_bracket_without_comma_string_to_list(x)
            if isinstance(x, str) else []
        )
    return buy, add, remove, visit, search, prod


def map_entities(buy, add, remove, visit, search, prod):
    """Map raw entity IDs to contiguous node indices."""
    user_ids = pd.concat([buy.client_id, add.client_id,
                           remove.client_id, visit.client_id,
                           search.client_id]).unique()
    item_ids = pd.concat([buy.sku, add.sku, remove.sku, prod.sku]).unique()
    url_ids  = visit.url.unique()
    cat_ids  = prod.category.unique()
    query_tuples = tuple(tuple(q) for q in search.query.values)
    query_ids= np.unique(list(query_tuples)) if len(query_tuples)>0 else []

    user2nid = {u: i for i, u in enumerate(user_ids)}
    item2nid = {i: i for i, i in enumerate(item_ids)}
    url2nid  = {u: i for i, u in enumerate(url_ids)}
    cat2nid  = {c: i for i, c in enumerate(cat_ids)}
    query2nid= {q: i for i, q in enumerate(query_tuples)}

    counts = {
        'num_users': len(user2nid),
        'num_items': len(item2nid),
        'num_urls': len(url2nid),
        'num_cats': len(cat2nid),
        'num_queries': len(query2nid)
    }
    return user2nid, item2nid, url2nid, cat2nid, query2nid, counts


def build_edges(buy, add, remove, visit, search, prod, mappings):
    """Build edge index dict for HeteroData."""
    user2nid, item2nid, url2nid, cat2nid, query2nid, _ = mappings
    edges = {}
    def map_ids(series, mapping):
        return series.map(mapping).values
    edges[('user','buys','item')]      = (map_ids(buy.client_id, user2nid),      map_ids(buy.sku, item2nid))
    edges[('user','adds','item')]      = (map_ids(add.client_id, user2nid),      map_ids(add.sku, item2nid))
    edges[('user','removes','item')]   = (map_ids(remove.client_id, user2nid),   map_ids(remove.sku, item2nid))
    edges[('user','visits','url')]     = (map_ids(visit.client_id, user2nid),    map_ids(visit.url, url2nid))
    edges[('item','is_in','category')] = (map_ids(prod.sku, item2nid),           map_ids(prod.category, cat2nid))
    edges[('category','is_of','item')] = (map_ids(prod.category, cat2nid),      map_ids(prod.sku, item2nid))
    edges[('url','is_visited_by','user')] = (map_ids(visit.url, url2nid),         map_ids(visit.client_id, user2nid))
    edges[('item','is_bought_by','user')] = (map_ids(buy.sku, item2nid),          map_ids(buy.client_id, user2nid))
    edges[('item','is_added_by','user')]   = (map_ids(add.sku, item2nid),          map_ids(add.client_id, user2nid))
    edges[('item','is_removed_by','user')] = (map_ids(remove.sku, item2nid),       map_ids(remove.client_id, user2nid))
    return edges


def build_hetero_graph(edges, counts):
    """Construct HeteroData with edge_index and node counts."""
    data = HeteroData()
    for (src, rel, dst), (src_idx, dst_idx) in edges.items():
        data[(src,rel,dst)].edge_index = torch.tensor([src_idx,dst_idx], dtype=torch.long)
    data['user'].num_nodes     = counts['num_users']
    data['item'].num_nodes     = counts['num_items']
    data['url'].num_nodes      = counts['num_urls']
    data['category'].num_nodes = counts['num_cats']
    # query node optional
    return data


def add_node_features(data, buy, add, remove, visit, search, prod, mappings, counts):
    """Add node features (price,name,search_emb,time_feats) to HeteroData."""
    user2nid, item2nid, url2nid, cat2nid, query2nid, _ = mappings
    # item features
    item_feat = prod.set_index('sku').loc[list(item2nid.keys())]
    data['item'].price = torch.tensor(item_feat.price.values, dtype=torch.long)
    data['item'].name  = torch.tensor(np.stack(item_feat.name.values), dtype=torch.long)
    # user search emb
    if len(search)>0:
        embed_dim = len(search.query.iloc[0])
        user_search = search.groupby('client_id').query.apply(lambda vs: np.mean(np.stack(vs.values), axis=0))
        search_list = user_search.reindex(user2nid.keys(), fill_value=[0]*embed_dim).tolist()
        data['user'].search_emb = torch.tensor(np.stack(search_list), dtype=torch.float)
    # time features
    windows=[1,7,30]
    max_time = max(buy.timestamp.max(), add.timestamp.max(), remove.timestamp.max())
    def build_time_feats(df, topk):
        days = (max_time-df.timestamp).dt.days
        df2 = df[days<=windows[-1]][['client_id','sku']].copy()
        df2['window']=pd.cut(days[days<=windows[-1]], bins=[-1]+windows, labels=windows).astype(int)
        df2 = df2[df2.sku.isin(topk)]
        grp = df2.groupby(['client_id','sku','window']).size()
        un = grp.unstack('window').reindex(
            index=pd.MultiIndex.from_product([user2nid.keys(), topk]),
            fill_value=0
        )
        arr = un.values.reshape(counts['num_users'], len(topk), len(windows))
        return arr.mean(axis=2)
    topk_buy = buy.sku.value_counts().index[:16]
    topk_add = add.sku.value_counts().index[:16]
    buy_feat = build_time_feats(buy, topk_buy)
    add_feat = build_time_feats(add, topk_add)
    time_feats = np.concatenate([buy_feat, add_feat], axis=1)
    data['user'].time_window_feats = torch.tensor(time_feats, dtype=torch.long)
    # final x
    if hasattr(data['user'], 'search_emb'):
        x = np.concatenate([time_feats, search_list], axis=1)
        data['user'].x = torch.tensor(x, dtype=torch.float)
    else:
        data['user'].x = torch.tensor(time_feats, dtype=torch.long)
    return data
