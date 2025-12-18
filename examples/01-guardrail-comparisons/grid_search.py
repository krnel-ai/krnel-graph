#!/usr/bin/env -S uv run

# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import pandas as pd
from main import *
from rich import print
from tqdm.auto import tqdm

## Derivative experiments
grid_searches = [
    probe_result.subs(probe, model_type="rbf_svc", params={"C": C})
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
] + [
    probe_result.subs(probe, model_type="logistic_regression", params={"C": C})
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
] + [
    probe_result.subs(probe, model_type="rbf_nusvm", params={"nu": nu})
    for nu in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
] + [
    probe_result.subs(probe, model_type="passive_aggressive")
]

if __name__ == "__main__":
    print("LlamaGuard evaluation:")
    print(llamaguard_result.to_json())

    print("\n\nKrnel Probe grid searches")
    metric = lambda g, name: g.to_json()["test"][name] if g.score.has_result() else None
    def better_factor(g, name):
        p = metric(g, name)
        q = llamaguard_result.to_json()["test"][name]
        if p is not None and q is not None:
            return (1.0 - q) / (1.0 - p)
    df = pd.DataFrame(
        [
            {
                "uuid": g.uuid,
                "precision@0.99": metric(g, "precision@0.99"),
                "better@0.99": better_factor(g, "precision@0.99"),
                "precision@0.999": metric(g, "precision@0.999"),
                "better@0.999": better_factor(g, "precision@0.999"),
                "model_type": g.score.model.model_type,
                "params": g.score.model.params,
            }
            for g in tqdm(grid_searches, leave=False)
        ]
    ).set_index("uuid").sort_values(by="precision@0.99", ascending=False)
    print(df)
