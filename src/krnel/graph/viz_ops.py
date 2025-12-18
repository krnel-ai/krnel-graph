# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any

from krnel.graph.types import VectorColumnType, VizEmbeddingColumnType


class UMAPVizOp(VizEmbeddingColumnType):
    """
    Compute a UMAP embedding, courtesy of UMAP-learn
    """

    input_embedding: VectorColumnType
    n_neighbors: int
    n_epochs: int
    random_state: int
    # n_components: int # only 2 supported

    # various knobs and dials

    metric: str = "euclidean"
    metric_kwds: dict[str, Any] | None = None
    output_metric: str = "euclidean"
    output_metric_kwds: dict[str, Any] | None = None
    learning_rate: float = 1.0
    min_dist: float = 0.1
    spread: float = 1.0
    set_op_mix_ratio: float = 1.0
    local_connectivity: float = 1.0
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    transform_queue_size: float = 4.0
    angular_rp_forest: bool = False
    target_n_neighbors: int = -1
    target_metric: str = "categorical"
    target_metric_kwds: dict[str, Any] | None = None
    target_weight: float = 0.5
    transform_seed: int = 42
    transform_mode: str = "embedding"
    force_approximation_algorithm: bool = False
    # verbose: bool = False
    # tqdm_kwds: dict[str, Any] | None = None
    unique: bool = False
    densmap: bool = False
    dens_lambda: float = 2.0
    dens_frac: float = 0.3
    dens_var_shift: float = 0.1
    output_dens: bool = False
    disconnection_distance: float | None = None
    # precomputed_knn: tuple[None | int, None | int, None | int] = (None, None, None),

    def __repr_html_runner__(
        self,
        runner: Any,
        color=None,
        label=None,
        scatter_kwargs=None,
        do_show: bool = True,
        **other_cols,
    ) -> Any:
        import jscatter
        import numpy as np
        import pandas as pd

        from krnel.graph.op_spec import OpSpec

        def to_np(value):
            if isinstance(value, OpSpec):
                value = runner.to_numpy(value)
            if isinstance(value, list):
                value = np.array(value)
            dtype = getattr(value, "dtype", None)
            if dtype is not None and np.issubdtype(dtype, np.bool_):
                value = np.array(["false", "true"])[value.astype(np.int8)]
            return value

        arr = to_np(self)
        df: dict[str, Any] = {"x": arr[:, 0], "y": arr[:, 1]}

        if color is not None:
            color = to_np(color)
            df["color"] = color
        if label is not None:
            label = to_np(label)
            df["label"] = label

        do_tooltip = False
        for name, column in other_cols.items():
            column = to_np(column)
            df[name] = column
            do_tooltip = True

        plot = jscatter.Scatter(
            data=pd.DataFrame(df),
            x="x",
            y="y",
            height=800,
            **(scatter_kwargs or {}),
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
