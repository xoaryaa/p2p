from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml
import networkx as nx
import os, json, re

from .ir import Graph

def _topo_order(g: Graph) -> List[str]:
    nxg = nx.DiGraph()
    nxg.add_nodes_from([n.id for n in g.nodes])
    for e in g.edges:
        nxg.add_edge(e.source, e.target)
    return list(nx.topological_sort(nxg))

def _load_graph(path: Path) -> Graph:
    data = yaml.safe_load(path.read_text())
    return Graph(**data)

def _read_text_fallback(text: Optional[str], text_file: Optional[Path]) -> str:
    if text is not None:
        return text
    if text_file is not None and text_file.exists():
        return text_file.read_text()
    return "Apple is opening a new office in Mumbai next year. Tim Cook met Prime Minister Modi in New Delhi."

def _read_query_fallback(query: Optional[str]) -> str:
    return query or "What is Apple doing in Mumbai?"

_word_re = re.compile(r"[A-Za-z']+")

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = _word_re.findall(text)
    out = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        out.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return out

class _BM25Index:
    def __init__(self, texts: List[str]):
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        self.texts = texts
        self.model = BM25Okapi(tokenized)
        self.tokenized = tokenized

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        scores = self.model.get_scores(query.lower().split())
        idx_scores = list(enumerate(scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        return idx_scores[:top_k]

def _load_docs_from_folder(path: Path) -> List[Tuple[str, str]]:
    docs = []
    path = Path(path)
    if not path.exists():
        return docs

    # PDFs
    for p in sorted(path.glob("*.pdf")):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            text = "".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                docs.append((p.name, text))
        except Exception as e:
            print(f"[runner] Warning: could not read PDF {p.name}: {e}")

    # TXTs
    for p in sorted(path.glob("*.txt")):
        try:
            docs.append((p.name, p.read_text()))
        except Exception as e:
            print(f"[runner] Warning: could not read TXT {p.name}: {e}")

    return docs

def run_graph(file: Path, *, text: Optional[str] = None, text_file: Optional[Path] = None,
              query: Optional[str] = None, docs_path: Optional[Path] = None, top_k: int = 3) -> bool:
    try:
        g = _load_graph(file)
    except Exception as e:
        print(f"[runner] Failed to load graph: {e}")
        return False

    values: Dict[str, Dict[str, Any]] = {}
    order = _topo_order(g)
    node_map = {n.id: n for n in g.nodes}

    for nid in order:
        node = node_map[nid]
        comp = node.component

        if comp == "InputText":
            txt = _read_text_fallback(text, text_file)
            values[nid] = {"text": txt}

        elif comp == "SpaCyModel":
            inbound_text = None
            for e in g.edges:
                if e.target == nid and e.target_input == "text":
                    inbound_text = values[e.source].get(e.source_output)
                    break
            if inbound_text is None:
                print(f"[runner] No inbound text for SpaCyModel at node '{nid}'.")
                return False

            model_name = node.params.get("model", "en_core_web_sm")
            try:
                import spacy
            except Exception as ie:
                print("[runner] spaCy not installed. Try: pip install spacy && python -m spacy download en_core_web_sm")
                print(f"[runner] Underlying error: {ie}")
                return False
            try:
                nlp = spacy.load(model_name)
            except Exception as me:
                print(f"[runner] Could not load spaCy model '{model_name}'. Install with: python -m spacy download en_core_web_sm")
                print(f"         Underlying error: {me}")
                return False

            doc = nlp(inbound_text)
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            values[nid] = {"doc": doc, "ents": ents}

        elif comp == "ConsolePrinter":
            inbound = None
            for e in g.edges:
                if e.target == nid:
                    inbound = values[e.source].get(e.source_output)
                    break
            if inbound is None:
                print(f"[runner] ConsolePrinter at '{nid}' has no inbound data.")
                return False

            print("=== ConsolePrinter ===")
            if isinstance(inbound, list):
                for i, it in enumerate(inbound, 1):
                    print(f"{i:02d}. {it}")
            else:
                print(inbound)
            values[nid] = {}

        elif comp == "PDFLoader":
            folder = Path(node.params.get("path", docs_path or "data/docs"))
            docs = _load_docs_from_folder(folder)
            if not docs:
                print(f"[runner] No documents found in '{folder}'. Place .pdf or .txt files there.")
            values[nid] = {"docs": docs}

        elif comp == "TextSplitter":
            inbound_docs = None
            for e in g.edges:
                if e.target == nid and e.target_input == "docs":
                    inbound_docs = values[e.source].get(e.source_output)
                    break
            if inbound_docs is None:
                print(f"[runner] TextSplitter at '{nid}' missing inbound docs.")
                return False
            chunk_size = int(node.params.get("chunk_size", 512))
            overlap = int(node.params.get("overlap", 64))
            chunks = []
            for name, text in inbound_docs:
                pieces = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for i, ch in enumerate(pieces):
                    chunks.append({"doc": name, "chunk_id": i, "text": ch})
            values[nid] = {"chunks": chunks}

        elif comp == "BM25Index":
            inbound_chunks = None
            for e in g.edges:
                if e.target == nid and e.target_input == "docs":
                    inbound_chunks = values[e.source].get(e.source_output)
                    break
            if inbound_chunks is None:
                print(f"[runner] BM25Index at '{nid}' missing inbound chunks.")
                return False
            texts = [c["text"] for c in inbound_chunks]
            index = _BM25Index(texts)
            values[nid] = {"index": index, "chunks": inbound_chunks}

        elif comp == "InputQuery":
            q = _read_query_fallback(query)
            values[nid] = {"query": q}

        elif comp == "BM25Retriever":
            inbound_query = None
            inbound_index = None
            inbound_chunks = None
            for e in g.edges:
                if e.target == nid and e.target_input == "query":
                    inbound_query = values[e.source].get(e.source_output)
                if e.target == nid and e.target_input == "index":
                    inbound_index = values[e.source].get(e.source_output)
                    if "chunks" in values[e.source]:
                        inbound_chunks = values[e.source]["chunks"]
            if inbound_query is None or inbound_index is None:
                print(f"[runner] BM25Retriever at '{nid}' missing query or index.")
                return False
            tk = int(node.params.get("top_k", top_k))
            hits_idx = inbound_index.search(inbound_query, top_k=tk)
            hits = []
            for idx, score in hits_idx:
                ch = inbound_chunks[idx]
                hits.append({"text": ch["text"], "score": float(score), "doc": ch["doc"], "chunk_id": ch["chunk_id"]})
            values[nid] = {"hits": hits}

        elif comp == "LLMReader":
            inbound_ctx = None
            inbound_q = None
            for e in g.edges:
                if e.target == nid and e.target_input == "context":
                    inbound_ctx = values[e.source].get(e.source_output)
                if e.target == nid and e.target_input == "question":
                    inbound_q = values[e.source].get(e.source_output)
            if not inbound_ctx:
                print(f"[runner] LLMReader at '{nid}' missing context.")
                return False
            top = inbound_ctx[0]
            answer = f"Top passage from {top['doc']} (chunk {top['chunk_id']}):\n" + top["text"]
            if inbound_q:
                answer = "Q: " + inbound_q + "\n" + answer
            values[nid] = {"answer": answer}

        elif comp == "ConsoleJSONWriter":
            inbound = None
            for e in g.edges:
                if e.target == nid:
                    inbound = values[e.source].get(e.source_output)
                    break
            if inbound is None:
                print(f"[runner] ConsoleJSONWriter at '{nid}' has no inbound data.")
                return False
            os.makedirs("artifacts", exist_ok=True)
            out_path = Path("artifacts") / f"{nid}.json"
            with open(out_path, "w") as f:
                json.dump(inbound, f, indent=2)
            print(f"[runner] Wrote JSON to {out_path}")
            values[nid] = {}

        else:
            print(f"[runner] Component '{comp}' not implemented yet. Skipping node '{nid}'.")
            values[nid] = {}

    print("[runner] Done.")
    return True
