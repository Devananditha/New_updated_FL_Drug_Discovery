"""Partition the global BioSNAP graph into simulated lab client graphs."""

import math
import random
from pathlib import Path

import networkx as nx

from graph_builder import load_biosnap_graph


def partition_graph(global_graph, num_clients=3):
    """Split graph edges into reproducible client subgraphs."""
    edges = list(global_graph.edges())

    random.seed(42)
    random.shuffle(edges)

    chunk_size = math.ceil(len(edges) / num_clients)
    partitions = []

    for client_index in range(num_clients):
        start = client_index * chunk_size
        end = start + chunk_size
        edge_chunk = edges[start:end]

        client_graph = nx.Graph()
        client_graph.add_edges_from(edge_chunk)
        partitions.append(client_graph)

    return partitions


def save_partitions(partitions, output_dir="data"):
    """Save each client graph partition as GraphML."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for index, graph in enumerate(partitions, start=1):
        filepath = output_path / f"client_{index}_graph.graphml"
        nx.write_graphml(graph, filepath)


if __name__ == "__main__":
    global_biosnap_graph = load_biosnap_graph()
    client_partitions = partition_graph(global_biosnap_graph)
    save_partitions(client_partitions)

    for index, graph in enumerate(client_partitions, start=1):
        print(
            f"Success: saved Client {index} graph with "
            f"{graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
        )
