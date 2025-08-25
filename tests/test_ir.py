from pathlib import Path
from prompt2pipes.generator import generate_graph_from_task, save_graph_yaml
from prompt2pipes.validator import validate_graph_from_file

def test_generate_and_validate(tmp_path: Path):
    g = generate_graph_from_task("ner")
    path = tmp_path / "ner.yaml"
    save_graph_yaml(g, path)
    ok, messages = validate_graph_from_file(path)
    assert ok, messages
