# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import html
import json
import uuid

_TEMPLATE = """
---
config:
  nodeSpacing: 1
  nodePadding: 1
  layout: elk
---
flowchart RL
{nodes}
{edges}
"""


class FlowchartReprMixin:
    # def _repr_html_(self):
    #     nodes = []
    #     edges = []
    #     for node in self.get_dependencies(recursive=True) + [self]:
    #         nodes.append(node._repr_flowchart_node_())
    #         edges.extend(list(node._repr_flowchart_edges_()))
    #     return _TEMPLATE.format(nodes="\n".join(nodes), edges="\n".join(edges))

    def _repr_mimebundle_(self, include=None, exclude=None):
        nodes = []
        edges = []
        for node in self.get_dependencies(recursive=True) + [self]:
            nodes.append(node._repr_flowchart_node_())
            edges.extend(list(node._repr_flowchart_edges_()))

        elem_id = f"mermaid-{uuid.uuid4().hex}"
        mermaid_content = _TEMPLATE.format(
            nodes=html.escape("\n".join(nodes)),
            edges=html.escape("\n".join(edges)),
        )
        html_bundle = f"""
        <div id="{elem_id}">
            <pre>{repr(self)}</pre>
        </div>
        <script>
        (async () => {{
          console.log("HELLO", document, window);
          const el = document.getElementById("{elem_id}");
          el.innerHTML = {json.dumps(mermaid_content)};

          const mod = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
          let mermaid = mod.default || mod;

          try {{
            mermaid.initialize({{ startOnLoad: false }});
            await mermaid.run({{ nodes: [el] }});
          }} catch (err) {{
            console.error("mermaid render error:", err);
          }}
        }})();
        </script>
        """
        return {
            "text/html": html_bundle,
            # "application/javascript": js,
        }


class FlowchartBigNode:
    def _repr_flowchart_node_(self):
        results = []
        results.append(f"subgraph {self._code_repr_identifier()}")
        results.append("  direction TB")
        for name, _dep in self.get_dependencies(include_names=True):
            results.append(
                f'  {self._code_repr_identifier()}_{name}@{{shape: "text", label: "{name}"}}'
            )
        results.append("end")
        return "\n".join(results)

    def _repr_flowchart_edges_(self):
        for name, dep in self.get_dependencies(include_names=True):
            yield f"{dep._code_repr_identifier()} --> {self._code_repr_identifier()}_{name}"
