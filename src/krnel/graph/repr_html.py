# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

_TEMPLATE = """
```mermaid
flowchart RL
{nodes}
{edges}
```"""

class FlowchartReprMixin:
    def _repr_markdown_(self):
        nodes = []
        edges = []
        for node in self.get_dependencies(recursive=True) + [self]:
            nodes.append(node._repr_flowchart_node_())
            edges.extend(list(node._repr_flowchart_edges_()))
        return _TEMPLATE.format(nodes="\n".join(nodes), edges="\n".join(edges))