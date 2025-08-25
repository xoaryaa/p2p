from pathlib import Path
import yaml
import networkx as nx
from .ir import Graph

def ascii_plan(path: Path) -> str:
    data = yaml.safe_load(path.read_text())
    g = Graph(**data)
    nxg = nx.DiGraph()
    nxg.add_nodes_from([n.id for n in g.nodes])
    for e in g.edges:
        nxg.add_edge(e.source, e.target, label=f"{e.source_output}->{e.target_input}")

    order = list(nx.topological_sort(nxg))
    lines = ["# ASCII Plan (topological order)"]
    for i, nid in enumerate(order, 1):
        node = next(n for n in g.nodes if n.id == nid)
        lines.append(f"{i:02d}. {node.id} [{node.component}]")
        for succ in nxg.successors(nid):
            elabel = nxg.get_edge_data(nid, succ)['label']
            lines.append(f"    └─▶ {succ}  ({elabel})")
    return "\n".join(lines)
