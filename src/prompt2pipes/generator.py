from importlib.resources import files
from pathlib import Path
import yaml
from .ir import Graph

def _load_template_yaml(name: str) -> str:
    pkg = files('prompt2pipes.templates')
    return (pkg / f"{name}.yaml").read_text()

def generate_graph_from_task(task: str) -> Graph:
    task = task.lower()
    if task not in {"ner", "rag-bm25"}:
        raise ValueError(f"Unknown task '{task}'. Use one of: ner, rag-bm25")
    data = yaml.safe_load(_load_template_yaml(task.replace('-', '_')))
    return Graph(**data)

def save_graph_yaml(graph: Graph, path: Path):
    path.write_text(yaml.safe_dump(graph.model_dump(), sort_keys=False))
