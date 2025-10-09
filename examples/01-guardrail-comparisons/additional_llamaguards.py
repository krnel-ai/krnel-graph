#!/usr/bin/env -S uv run

# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from main import *

additional_llamaguards = [
    llamaguard_result.subs(
        llamaguard_scores,
        model_name="hf:meta-llama/Llama-Guard-3-8B",
        logit_token_ids=["safe", "unsafe"],
        append_to_chat_template="\n\n\n",
    ),
]

