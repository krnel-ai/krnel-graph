# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai


import numpy as np
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from sklearn import calibration

from krnel.graph.classifier_ops import ClassifierPredictOp, TrainClassifierOp
from krnel.graph.runners.local_runner.local_arrow_runner import LocalArrowRunner
from krnel.logging import get_logger

logger = get_logger(__name__)

_MODELS = {
    "logistic_regression": lambda kw: sklearn.linear_model.LogisticRegression(**kw),
    "linear_svc": lambda kw: sklearn.svm.LinearSVC(**kw),
    "passive_aggressive": lambda kw: sklearn.linear_model.PassiveAggressiveClassifier(
        **kw
    ),
    "rbf_nusvm": lambda kw: sklearn.svm.NuSVC(kernel="rbf", **kw),
    "rbf_svc": lambda kw: sklearn.svm.SVC(kernel="rbf", **kw),
    "calibrated_rbf_nusvm": lambda kw: calibration.CalibratedClassifierCV(
        sklearn.svm.NuSVC(kernel="rbf", **kw),
        cv=5,
    ),
}


@LocalArrowRunner.implementation
def train_model(runner, op: TrainClassifierOp):
    log = logger.bind(op=op.uuid)
    x = runner.to_numpy(op.x).astype("float32")
    positives = runner.to_numpy(op.positives)
    if positives.dtype != np.bool_:
        raise TypeError(f"Expected bool dtype for positives, got {positives.dtype}")
    negatives = runner.to_numpy(op.negatives)
    if negatives.dtype != np.bool_:
        raise TypeError(f"Expected bool dtype for negatives, got {negatives.dtype}")
    if positives.sum() == 0:
        raise ValueError("No positive samples found")
    if negatives.sum() == 0:
        raise ValueError("No negative samples found")
    if (n_inconsistent := (positives & negatives).sum()) > 0:
        raise ValueError(
            f"Some examples ({n_inconsistent}) are both positive and negative"
        )

    mask = positives | negatives

    if op.train_domain is not None:
        train_domain = runner.to_numpy(op.train_domain)
        if train_domain.dtype != np.bool_:
            raise TypeError(
                f"Expected bool dtype for train_domain, got {train_domain.dtype}"
            )
        log = log.bind(orig_domain=len(train_domain), train_domain=train_domain.sum())
        mask = mask & train_domain

    x = x[mask]
    positives = positives[mask]
    negatives = negatives[mask]

    model = _MODELS[op.model_type](op.params)

    match op.preprocessing:
        case "none":
            pass
        case "standardize":
            model = sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.StandardScaler(), model
            )
        case "normalize":
            model = sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.Normalizer(), model
            )

    log.info(
        "Fitting estimator",
        model=model,
        x_shape=x.shape,
        y_shape=positives.shape,
        n_positives=positives.sum(),
        n_negatives=negatives.sum(),
        dtype=x.dtype,
    )
    model.fit(x, positives)
    runner.write_sklearn_estimator(op, model)


@LocalArrowRunner.implementation
def decision_function(runner, op: ClassifierPredictOp):
    log = logger.bind(op=op.uuid)
    x = runner.to_numpy(op.x).astype("float32")
    clsf = runner.to_sklearn_estimator(op.model)
    log.info("Computing decision function", model=clsf)

    if hasattr(clsf, "predict_proba"):
        p = clsf.predict_proba(x)
        if p.ndim == 2 and p.shape[1] == 2:
            result = p[:, 1]
        else:
            raise ValueError(f"Multiclass not implemented. Shape: {p.shape}")
    if hasattr(clsf, "decision_function"):
        result = clsf.decision_function(x)
    else:
        raise NotImplementedError(f"Not sure how to get scores from {clsf}")
    runner.write_numpy(op, result)
