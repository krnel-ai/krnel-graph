# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import difflib

from krnel.graph.op_spec import OpSpec


class GraphDiff:
    def __init__(self, a: OpSpec, b: OpSpec):
        self.a = a
        self.b = b

    def __repr__(self):
        """
        Compute the diff between two OpSpec instances.
        """
        return "\n".join(
            difflib.unified_diff(
                self.a.to_code().splitlines(),
                self.b.to_code().splitlines(),
                fromfile=self.a.uuid,
                tofile=self.b.uuid,
            )
        )

    def _repr_html_(self):
        """
        Compute the HTML diff between two OpSpec instances.
        """
        styles = """
        <style>
        table.diff {font-family:Courier; border:medium;}
        table.diff td {text-align:left; vertical-align:top; max-width: 600px; white-space: pre-wrap; word-wrap: break-word; margin: 0; padding: 0;}
        .diff_header {background-color:#f0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#d0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
        </style>
        """
        return styles + (
            difflib.HtmlDiff().make_table(
                self.a.to_code().splitlines(),
                self.b.to_code().splitlines(),
                fromdesc=self.a.uuid,
                todesc=self.b.uuid,
            )
        )
