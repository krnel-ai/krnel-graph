# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai


from collections import defaultdict
import numpy as np
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from sklearn import calibration

from krnel.graph.classifier_ops import ClassifierEvaluationOp, ClassifierPredictOp, TrainClassifierOp
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


@LocalArrowRunner.implementation
def evaluate_scores(runner, op: ClassifierEvaluationOp):
    """Evaluate classification scores."""
    from sklearn import metrics

    log = logger.bind(op=op.uuid)
    scores = runner.to_numpy(op.score)

    gt_positives = runner.to_numpy(op.gt_positives)
    if gt_positives.dtype != np.bool_:
        raise TypeError(
            f"Expected bool dtype for gt_positives, got {gt_positives.dtype}"
        )
    gt_negatives = runner.to_numpy(op.gt_negatives)
    if gt_negatives.dtype != np.bool_:
        raise TypeError(
            f"Expected bool dtype for gt_negatives, got {gt_negatives.dtype}"
        )
    if (n_inconsistent := (gt_positives & gt_negatives).sum()) > 0:
        raise ValueError(
            f"Some examples ({n_inconsistent}) are both positive and negative"
        )

    per_split_metrics = defaultdict(dict)

    def compute_classification_metrics(y_true, y_score):
        """Appropriate for binary classification results."""
        result = {}
        result["count"] = len(y_true)
        result["n_true"] = int(y_true.sum())
        prec, rec, thresh = metrics.precision_recall_curve(y_true, y_score)
        # result[f"pr_curve"] = {
        #    "precision": prec.tolist(),
        #    "recall": rec.tolist(),
        #    "threshold": thresh.tolist(),
        # }
        roc_fpr, roc_tpr, roc_thresh = metrics.roc_curve(y_true, y_score)
        # result["roc_curve"] = metrics.roc_curve(y_true, y_score)
        result["average_precision"] = metrics.average_precision_score(y_true, y_score)
        result["roc_auc"] = metrics.roc_auc_score(y_true, y_score)

        # pick best score for accuracy
        for _, _, thresh in zip(prec, rec, np.append(thresh, 1.0)):
            y_pred = y_score >= thresh
            acc = (y_pred == y_true).mean()
            if "accuracy" not in result or acc > result["accuracy"]:
                result["accuracy"] = acc
                result["confusion"] = {
                    "tp": int((y_pred & y_true).sum()),
                    "fp": int((y_pred & ~y_true).sum()),
                    "tn": int((~y_pred & y_true).sum()),
                    "fn": int((~y_pred & ~y_true).sum()),
                }

        for recall in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
            precision = prec[rec >= recall].max()
            if np.isnan(precision):
                precision = 0.0
            result[f"precision@{recall}"] = precision
        return result

    splits = None
    if op.split is not None:
        splits = runner.to_numpy(op.split)

    if splits is None:
        log.debug("No splits provided, grouping all samples into one 'all' split")
        splits = np.array(["all"] * len(scores))

    domain = None
    if op.predict_domain is not None:
        domain = runner.to_numpy(op.predict_domain)
        if domain.dtype != np.bool_:
            raise TypeError(f"Expected bool dtype for domain, got {domain.dtype}")

    if domain is None:
        log.debug("No domain provided, using all samples")
        domain = np.array([True] * len(scores))

    for split in set(splits):
        split_mask = (splits == split) & domain & (gt_positives | gt_negatives)
        per_split_metrics[split] = compute_classification_metrics(
            gt_positives[split_mask], scores[split_mask]
        )

    runner.write_json(op, dict(per_split_metrics))
