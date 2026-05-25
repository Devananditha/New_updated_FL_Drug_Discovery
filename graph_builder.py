"""Load BioSNAP drug-target edges into a NetworkX graph."""

from pathlib import Path

import networkx as nx
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "ChG-TargetDecagon_targets.csv.gz"
)


def load_biosnap_graph(filepath: str | Path = DEFAULT_DATA_PATH) -> nx.Graph:
    """Read compressed BioSNAP edge list and return an undirected graph."""
    df = pd.read_csv(
        filepath,
        compression="gzip",
        header=None,
        skiprows=1,  # skip "# Drug\tGene" comment header
        names=["source", "target"],
    )
    return nx.from_pandas_edgelist(df, source="source", target="target")


if __name__ == "__main__":
    graph = load_biosnap_graph()
    print(
        f"Success: loaded BioSNAP graph with "
        f"{graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
    )
