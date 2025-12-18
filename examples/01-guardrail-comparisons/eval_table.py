#!/usr/bin/env -S uv run
# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from main import *
from rich import print
import pandas as pd

if __name__ == "__main__":
    print("LlamaGuard evaluation:")
    print(llamaguard_result.to_json())
    print("\n\nKrnel Probe evaluation:")
    print(probe_result.to_json())

    print("\n\nConfusion matrices by data source:")

    def confusion_by_source_df(evaluation_op, label):
        result = evaluation_op.subs(
            split=col_source,              # group by data source
            predict_domain=col_split.test, # only evaluate on test set
        ).to_json()
        # only get confusion matrix and discard other metrics
        result = {split: result[split].get("confusion", {"tp":0,"fp":0,"tn":0,"fn":0}) for split in result}
        # convert to dataframe
        df = pd.DataFrame(result).T.sort_index()
        df.columns = pd.MultiIndex.from_product([[label], df.columns])
        return df

    combined_confusion = pd.concat(
        [
            confusion_by_source_df(llamaguard_result, "LlamaGuard"),
            confusion_by_source_df(probe_result, "Krnel Probe"),
        ],
        axis=1,
    )

    print(combined_confusion)