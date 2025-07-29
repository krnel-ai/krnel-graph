import numpy as np
from krnel.graph.viz_ops import UMAPVizOp

import jscatter
import pandas as pd

def umap_viz(runner, op: UMAPVizOp, color=None, label=None, **other_cols) -> str:
    def to_np(op):
        x = runner.materialize(op).to_numpy()
        if x.dtype == np.bool_:
            x = np.array(['true', 'false'])[x.astype(np.int8)]
        return x
    arr = to_np(op)
    df = {'x': arr[:, 0], 'y': arr[:, 1]}
    if color is not None:
        color = to_np(color)
        df['color'] = color
    if label is not None:
        label = to_np(label)
        df['label'] = label

    do_tooltip=False
    for name, col in other_cols.items():
        col = to_np(col)
        df[name] = col
        do_tooltip=True

    plot = jscatter.Scatter(data=pd.DataFrame(df), x='x', y='y', height=800)

    if color is not None:
        plot.color(by='color', legend=True)
        plot.legend(legend=True)
    if label is not None:
        plot.label(by='label')

    if do_tooltip:
        plot.tooltip(enable=True, properties=list(df.keys()), preview_text_lines=4)

    return plot.show()