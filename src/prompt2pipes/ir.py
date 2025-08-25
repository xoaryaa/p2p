from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class Node(BaseModel):
    id: str
    component: str
    inputs: Dict[str, str] = Field(default_factory=dict)   # name -> type
    outputs: Dict[str, str] = Field(default_factory=dict)  # name -> type
    params: Dict[str, Any] = Field(default_factory=dict)

class Edge(BaseModel):
    source: str
    source_output: str
    target: str
    target_input: str

class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def node_map(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def outputs_of(self, node_id: str) -> Dict[str, str]:
        return self.node_map()[node_id].outputs

    def inputs_of(self, node_id: str) -> Dict[str, str]:
        return self.node_map()[node_id].inputs
