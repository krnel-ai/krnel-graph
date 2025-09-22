# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

_TEMPLATE = """
```mermaid
---
config:
  nodeSpacing: 1
  padding: 1
---
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


class FlowchartBigNode:
    def _repr_flowchart_node_(self):
        results = []
        results.append(f"subgraph {self._code_repr_identifier()}")
        results.append("  direction TB")
        for name, dep in self.get_dependencies(include_names=True):
            results.append(
                f'  {self._code_repr_identifier()}_{name}@{{shape: "text", label: "{name}"}}'
            )
        results.append("end")
        return "\n".join(results)

    def _repr_flowchart_edges_(self):
        for name, dep in self.get_dependencies(include_names=True):
            yield f"{dep._code_repr_identifier()} --> {self._code_repr_identifier()}_{name}"
