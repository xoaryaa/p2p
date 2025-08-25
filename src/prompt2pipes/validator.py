from pathlib import Path
import yaml
import networkx as nx
from typing import Tuple, List
from .ir import Graph

def _load_graph(path: Path) -> Graph:
    data = yaml.safe_load(path.read_text())
    return Graph(**data)

def validate_graph_from_file(path: Path) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    ok = True
    g = _load_graph(path)

    node_ids = {n.id for n in g.nodes}
    # 1) Unique node ids
    if len(node_ids) != len(g.nodes):
        ok = False
        messages.append("ERR: Duplicate node IDs detected.")
    else:
        messages.append("OK: Node IDs are unique.")

    # 2) Edges refer to existing nodes
    for e in g.edges:
        if e.source not in node_ids or e.target not in node_ids:
            ok = False
            messages.append(f"ERR: Edge {e.source}->{e.target} references missing node(s)." )
    if ok:
        messages.append("OK: All edges reference existing nodes.")

    # 3) Outputs/inputs exist
    node_map = {n.id: n for n in g.nodes}
    for e in g.edges:
        so = node_map[e.source].outputs
        ti = node_map[e.target].inputs
        if e.source_output not in so:
            ok = False
            messages.append(f"ERR: Edge from {e.source}.{e.source_output} not an output on that node.")
        if e.target_input not in ti:
            ok = False
            messages.append(f"ERR: Edge to {e.target}.{e.target_input} not an input on that node.")
    if ok:
        messages.append("OK: All edge endpoints correspond to declared inputs/outputs.")

    # 4) Acyclic check
    nxg = nx.DiGraph()
    nxg.add_nodes_from(node_ids)
    for e in g.edges:
        nxg.add_edge(e.source, e.target)
    try:
        list(nx.topological_sort(nxg))
        messages.append("OK: Graph is acyclic.")
    except nx.NetworkXUnfeasible:
        ok = False
        messages.append("ERR: Cycle detected in the graph.")

    return ok, messages
