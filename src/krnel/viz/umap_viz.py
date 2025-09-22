# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import jscatter
import numpy as np

from krnel.graph.op_spec import OpSpec
from krnel.graph.viz_ops import UMAPVizOp


def umap_viz(
    runner,
    op: UMAPVizOp,
    color=None,
    label=None,
    scatter_kwargs=None,
    do_show=True,
    **other_cols,
) -> str:
    import pandas as pd

    def to_np(x):
        if isinstance(x, OpSpec):
            x = runner.to_numpy(x)
        if isinstance(x, list):
            x = np.array(x)
        if x.dtype == np.bool_:
            x = np.array(["false", "true"])[x.astype(np.int8)]
        return x

    arr = to_np(op)
    df = {"x": arr[:, 0], "y": arr[:, 1]}

    if color is not None:
        color = to_np(color)
        df["color"] = color
    if label is not None:
        label = to_np(label)
        df["label"] = label

    do_tooltip = False
    for name, col in other_cols.items():
        col = to_np(col)
        df[name] = col
        do_tooltip = True

    plot = jscatter.Scatter(
        data=pd.DataFrame(df), x="x", y="y", height=800, **(scatter_kwargs or {})
    )

    if color is not None:
        plot.color(by="color", legend=True)
        plot.legend(legend=True)
    if label is not None:
        plot.label(by="label")

    if do_tooltip:
        plot.tooltip(
            enable=True,
            properties=list(df.keys()),
            preview_text_lines=None,
            size="large",
        )

    if do_show:
        return plot.show()
    return plot
