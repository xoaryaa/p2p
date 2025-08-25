from pathlib import Path
import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from typing import Optional

from .generator import generate_graph_from_task, save_graph_yaml
from .validator import validate_graph_from_file
from .visualize import ascii_plan

app = typer.Typer(no_args_is_help=True, help="Prompt2Pipes CLI — natural language → NLP pipelines")

@app.command()
def init():
    """Create a local project layout (pipelines/, data/, artifacts/)."""
    for name in ["pipelines", "data", "artifacts"]:
        Path(name).mkdir(exist_ok=True)
    rprint(Panel.fit("[bold green]Initialized[/] directories: pipelines/, data/, artifacts/"))

@app.command()
def generate(task: str = typer.Option(..., help="Template to use: ner | rag-bm25"),
             name: str = typer.Option("pipeline", help="Output filename (without .yaml)"),
             outdir: Path = typer.Option(Path("pipelines"), help="Where to place the YAML"),
    ):
    """Generate a pipeline YAML from a known template."""
    graph = generate_graph_from_task(task)
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / f"{name}.yaml"
    save_graph_yaml(graph, outfile)
    rprint(Panel.fit(f"Saved template [bold]{task}[/] to [cyan]{outfile}[/]"))

@app.command()
def validate(file: Path):
    """Validate a pipeline YAML (structure, nodes/edges, cycles)."""
    ok, messages = validate_graph_from_file(file)
    table = Table(title="Validation Report", show_lines=True)
    table.add_column("Status", justify="center", style="bold")
    table.add_column("Message")
    for m in messages:
        status = "OK" if m.startswith("OK:") else "ERR"
        table.add_row(status, m)
    rprint(table)
    if not ok:
        raise typer.Exit(code=1)

@app.command()
def explain(file: Path):
    """Print an ASCII plan of the pipeline graph."""
    print(ascii_plan(file))

@app.command()
def run(file: Path,
        text: Optional[str] = typer.Option(None, help="Inline text input (NER)."),
        text_file: Optional[Path] = typer.Option(None, help="Path to a .txt file to use as input (NER)."),
        query: Optional[str] = typer.Option(None, help="Query text for RAG pipelines."),
        docs_path: Optional[Path] = typer.Option(Path("data/docs"), help="Folder with .pdf or .txt docs."),
        top_k: int = typer.Option(3, help="Retriever top-k.")):
    """Execute the pipeline (NER & RAG)."""
    from .runner import run_graph
    ok = run_graph(file, text=text, text_file=text_file, query=query, docs_path=docs_path, top_k=top_k)
    if not ok:
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
