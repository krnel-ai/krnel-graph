# Copyright (c) 2025 Krnel
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

    def __repr_html_runner__(self, runner: Any, **kwargs) -> str:
        from krnel.viz.umap_viz import umap_viz

        return umap_viz(runner, self, **kwargs)
