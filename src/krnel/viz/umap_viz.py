from krnel.graph.viz_ops import UMAPVizOp

import jscatter
import pandas as pd

def umap_viz(runner, op: UMAPVizOp, color=None, label=None, **other_cols) -> str:
    arr = runner.materialize(op, as_numpy=True)


    df = {'x': arr[:, 0], 'y': arr[:, 1]}
    if color is not None:
        color = runner.materialize(color, as_numpy=True)
        df['color'] = color
    if label is not None:
        label = runner.materialize(label, as_numpy=True)
        df['label'] = label

    do_tooltip=False
    for name, col in other_cols.items():
        col = runner.materialize(col, as_numpy=True)
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