# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import krnel.graph
from rich import print

runner = krnel.graph.Runner()
ds = runner.from_parquet(
    "/Users/kimmy/Downloads/krnel_harmful_20250204.parquet",
    sha256sum="d29aada58992822c86733d97eb629c1cc057e73af3fb6d959aa10c7c03230a12",
)
col_harmful = ds.col_boolean('harmful')
col_text = ds.col_text('prompt')
col_split = ds.assign_train_test_split()
col_source = ds.col_categorical('source')

######
# [Llamaguard](https://huggingface.co/meta-llama/LlamaGuard-7b)
# - Score: Difference between "unsafe" and "safe" token logits
# - Reference: https://arxiv.org/abs/2309.06161
llamaguard_scores = s = col_text.llm_logit_scores(
    model_name="hf:meta-llama/LlamaGuard-7b",
    batch_size=1,
    logit_token_ids=[9109, 25110], # ["_safe", "_unsafe"],
    apply_chat_template=True,
    dtype="float16",
    max_length=2048,
    torch_compile=True,
)
llamaguard_result = (
    # Difference of "unsafe" - "safe" logits
    llamaguard_scores.col(1) - llamaguard_scores.col(0)
).evaluate(gt_positives=col_harmful)

X = col_text.llm_layer_activations(
    model_name="hf:meta-llama/Llama-2-7b-chat-hf",
    layer_num=-1,
    token_mode="last",
    batch_size=4,
    max_length=2048,
    dtype="float16",
)
probe = X.train_classifier(
    "rbf_nusvm",
    positives=col_harmful,
    negatives=~col_harmful,
    train_domain=col_split.train,
    preprocessing="standardize",
)
probe_result = probe.predict(X).evaluate(gt_positives=col_harmful, split=col_split)


llamaguard3_result = llamaguard_result.subs(
    llamaguard_scores,
    model_name="hf:meta-llama/Llama-Guard-3-8B",
    logit_token_ids=["safe", "unsafe"],
    append_to_chat_template="\n\n\n",
)
# llamaguard4_result = llamaguard_result.subs(
#    llamaguard_scores,
#    model_name="hf:meta-llama/Llama-Guard-4-12B",
#    logit_token_ids=["_unsafe", "_safe"],
# )

if __name__ == "__main__":
    print("LlamaGuard evaluation:")
    print(runner.to_json(llamaguard_result))

    print("By source:")
    print(runner.to_json(llamaguard_result.subs(split=col_source)))
