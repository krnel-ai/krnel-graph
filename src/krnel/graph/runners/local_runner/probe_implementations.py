# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai


from collections import defaultdict
from typing import Any, Callable
import numpy as np
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.kernel_approximation
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from sklearn import calibration

from krnel.graph.classifier_ops import ClassifierEvaluationOp, ClassifierPredictOp, TrainClassifierOp
from krnel.graph.runners.local_runner.local_arrow_runner import LocalArrowRunner
from krnel.logging import get_logger

logger = get_logger(__name__)

__all__ = ["register_classifier_model", "get_classifier_model"]

# Global registry for classifier models
_CLASSIFIER_MODELS: dict[str, Callable[[dict], Any]] = {}


def register_classifier_model(name: str):
    """
    Decorator to register a classifier model factory function.

    The decorated function should accept a dict of parameters and return
    an sklearn-compatible classifier instance.

    Example:
        @register_classifier_model("my_classifier")
        def create_my_classifier(params):
            return MyClassifier(**params)
    """
    def decorator(factory_func: Callable[[dict], Any]) -> Callable[[dict], Any]:
        if name in _CLASSIFIER_MODELS:
            raise ValueError(f"Classifier model '{name}' is already registered")
        _CLASSIFIER_MODELS[name] = factory_func
        return factory_func
    return decorator


def get_classifier_model(name: str) -> Callable[[dict], Any]:
    """Get a classifier model factory by name."""
    if name not in _CLASSIFIER_MODELS:
        available = ", ".join(sorted(_CLASSIFIER_MODELS.keys()))
        raise ValueError(
            f"Unknown classifier model '{name}'. "
            f"Available models: {available}"
        )
    return _CLASSIFIER_MODELS[name]


@register_classifier_model("logistic_regression")
def _create_logistic_regression(params):
    return sklearn.linear_model.LogisticRegression(**params)


@register_classifier_model("linear_svc")
def _create_linear_svc(params):
    return sklearn.svm.LinearSVC(**params)


@register_classifier_model("passive_aggressive")
def _create_passive_aggressive(params):
    return sklearn.linear_model.PassiveAggressiveClassifier(**params)


@register_classifier_model("rbf_nusvm")
def _create_rbf_nusvm(params):
    return sklearn.svm.NuSVC(kernel="rbf", **params)


@register_classifier_model("rbf_svc")
def _create_rbf_svc(params):
    return sklearn.svm.SVC(kernel="rbf", **params)


@register_classifier_model("calibrated_rbf_nusvm")
def _create_calibrated_rbf_nusvm(params):
    cv = params.pop("cv", None)
    return calibration.CalibratedClassifierCV(
        sklearn.svm.NuSVC(kernel="rbf", **params),
        cv=cv,
    )

@register_classifier_model("sgd")
def _create_sgd(params):
    return sklearn.linear_model.SGDClassifier(**params)

@register_classifier_model("sgd_nystroem")
def _create_sgd_nystroem(params):
    return sklearn.pipeline.make_pipeline(
        sklearn.kernel_approximation.Nystroem(**params.get("nystroem_params", {})),
        sklearn.linear_model.SGDClassifier(**params.get("sgd_params", {})),
    )

@register_classifier_model("sgd_nystroem_one_class_svm")
def _create_sgd_nystroem_one_class_svm(params):
    return sklearn.pipeline.make_pipeline(
        sklearn.kernel_approximation.Nystroem(**params.get("nystroem_params", {})),
        sklearn.linear_model.SGDOneClassSVM(**params.get("sgd_params", {})),
    )

@register_classifier_model("random_forest")
def _create_random_forest(params):
    return sklearn.ensemble.RandomForestClassifier(**params)

@register_classifier_model("naive_bayes")
def _create_naive_bayes(params):
    return sklearn.naive_bayes.GaussianNB(**params)


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

    model_factory = get_classifier_model(op.model_type)
    model = model_factory(op.params)

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

    def compute_classification_metrics(split, y_true, y_score):
        """Appropriate for binary classification results."""
        def warn(msg):
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append(msg)
            #log.warning(f"{msg}", split=split)

        result = {}
        result["count"] = len(y_true)
        result["n_true"] = int(y_true.sum())
        result["n_false"] = int((1 - y_true).sum())
        result["avg_score"] = float(y_score.mean())
        if len(y_true) == 0:
            warn("No samples in this split")
            return result
        prec, rec, thresh = metrics.precision_recall_curve(y_true, y_score, drop_intermediate=True)
        # result[f"pr_curve"] = {
        #    "precision": prec.tolist(),
        #    "recall": rec.tolist(),
        #    "threshold": thresh.tolist(),
        # }
        #roc_fpr, roc_tpr, roc_thresh = metrics.roc_curve(y_true, y_score)

        if op.score_threshold is None:
            # pick best score for accuracy
            if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
                # all true or all false, accuracy is undefined
                warn(f"Accuracy not defined when all groundtruth labels are {y_true[0]}, set op.score_threshold")
            else:
                for thresh in np.append(thresh, 1.0):
                    y_pred = y_score >= thresh
                    acc = (y_pred == y_true).mean()
                    if "best_accuracy" not in result or acc > result["best_accuracy"]:
                        result["best_accuracy"] = acc
                        result["most_accurate_threshold"] = float(thresh)
                        result["best_confusion"] = {
                            "tn": int((~y_pred & ~y_true).sum()),
                            "fp": int((y_pred & ~y_true).sum()),
                            "fn": int((~y_pred & y_true).sum()),
                            "tp": int((y_pred & y_true).sum()),
                        }
        else:
            y_pred = y_score >= op.score_threshold
            acc = (y_pred == y_true).mean()
            result["accuracy"] = acc
            result["confusion"] = {
                "tn": int((~y_pred & ~y_true).sum()),
                "fp": int((y_pred & ~y_true).sum()),
                "fn": int((~y_pred & y_true).sum()),
                "tp": int((y_pred & y_true).sum()),
            }
            result["precision"] = metrics.precision_score(y_true, y_pred)
            result["recall"] = metrics.recall_score(y_true, y_pred)
            result["f1"] = metrics.f1_score(y_true, y_pred)

        if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
            warn(f"Precision/recall curve not defined when all groundtruth labels are {y_true[0]}")
        else:
            result["average_precision"] = metrics.average_precision_score(y_true, y_score)
            result["roc_auc"] = metrics.roc_auc_score(y_true, y_score)
            for recall in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
                # NaN if all groundtruth labels are positive or negative
                precision = prec[rec >= recall].max()
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
            split,
            gt_positives[split_mask], scores[split_mask]
        )

    runner.write_json(op, dict(per_split_metrics))
