# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# ruff: noqa: S101

import pytest
import sklearn.linear_model

from krnel.graph.runners.local_runner.probe_implementations import (
    register_classifier_model,
    get_classifier_model,
)


def test_all_builtin_models_registered():
    """Verify all 6 built-in models are accessible."""
    models = [
        "logistic_regression",
        "linear_svc",
        "passive_aggressive",
        "rbf_nusvm",
        "rbf_svc",
        "calibrated_rbf_nusvm",
    ]
    for model_name in models:
        factory = get_classifier_model(model_name)
        assert callable(factory)


def test_custom_model_registration():
    """Test that custom models can be registered."""

    @register_classifier_model("test_ridge")
    def create_ridge(params):
        return sklearn.linear_model.Ridge(**params)

    factory = get_classifier_model("test_ridge")
    model = factory({"alpha": 1.0})
    assert isinstance(model, sklearn.linear_model.Ridge)


def test_unknown_model_error():
    """Test error handling for unknown model names."""
    with pytest.raises(ValueError, match="Unknown classifier model"):
        get_classifier_model("nonexistent_model")


def test_duplicate_registration_error():
    """Test that registering the same model name twice raises an error."""

    @register_classifier_model("test_duplicate")
    def create_first(params):
        return sklearn.linear_model.Ridge(**params)

    with pytest.raises(ValueError, match="already registered"):
        @register_classifier_model("test_duplicate")
        def create_second(params):
            return sklearn.linear_model.Lasso(**params)


def test_model_factory_returns_correct_instance():
    """Test that model factories return the expected sklearn model types."""
    test_params = {}

    # Test logistic regression
    lr_factory = get_classifier_model("logistic_regression")
    lr_model = lr_factory(test_params)
    assert isinstance(lr_model, sklearn.linear_model.LogisticRegression)

    # Test linear SVC
    svc_factory = get_classifier_model("linear_svc")
    svc_model = svc_factory(test_params)
    assert isinstance(svc_model, sklearn.svm.LinearSVC)


def test_model_factory_accepts_parameters():
    """Test that model factories properly pass parameters to sklearn models."""
    lr_factory = get_classifier_model("logistic_regression")
    model = lr_factory({"max_iter": 500, "C": 0.5})

    assert model.max_iter == 500
    assert model.C == 0.5
