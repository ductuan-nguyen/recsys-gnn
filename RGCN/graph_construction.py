import os
import time
import torch
import argparse
from rgcn_final.graph_utils import (
    load_dataframes, map_entities, build_edges,
    build_hetero_graph, add_node_features
)

def parse_args():
    parser = argparse.ArgumentParser(description='Construct and save a heterogeneous graph')
    parser.add_argument('data_dir', type=str, help='Directory containing parquet input files')
    parser.add_argument('output_path', type=str, help='File path to save the HeteroData object')
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()
    # load raw dataframes
    buy, add, remove, visit, search, prod = load_dataframes(args.data_dir)
    # map entity IDs to node indices
    mappings, counts = map_entities(buy, add, remove, visit, search, prod)
    # build edge index dict
    edges = build_edges(buy, add, remove, visit, search, prod, mappings)
    # construct HeteroData
    data = build_hetero_graph(edges, counts)
    # add node features
    data = add_node_features(data, buy, add, remove, visit, search, prod, mappings, counts)
    # save graph
    torch.save(data, args.output_path)
    end = time.time()
    print(f"Hetero graph saved to {args.output_path}")
    print(f"Total time: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()