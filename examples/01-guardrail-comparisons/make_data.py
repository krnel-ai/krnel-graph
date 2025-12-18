#!/usr/bin/env -S uv run
# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

#   Guardrail datasets:
#     A mix of many sources.
#       Please see the README.
#                        - kjw

EXPECTED_SIZES = [
    # source                 harmful  count
    ('tatsu-lab-alpaca',       False, 52001),
    ('babelscape_alert_adv',   True,  30771),
    ('jailbreak_llms',         True,  19738),
    ('babelscape_alert',       True,  14092),
    ('jailbreak_llms',         False,  9638),
    ('sorrybench',             True,   9439),
    ('steering-toxic',         True,   7377),
    ('advbench',               True,    520),
    ('many-shot-jailbreaking', True,    266),
    ('GPTFuzz',                True,    100),
]

SQL = r"""
install zipfs FROM community;
load zipfs;

create secret hf_token (type huggingface, provider credential_chain);

create temporary table dataset as
select distinct prompt
    .regexp_replace('openai', 'AI designer', 'gi')
    .regexp_replace('chatgpt', 'AI', 'gi')
    .regexp_replace('### (Instruction|Response):', '', 'sgi')
    -- .trim(' \n\t\r')
    as prompt, harmful, source
from (
    select prompt, jailbreak harmful, 'jailbreak_llms' as source from 'hf://datasets/TrustAIRLab/in-the-wild-jailbreak-prompts/**/*.parquet'
    union
    select distinct prompt || e'.\n' || question as prompt, true as harmful, 'jailbreak_llms' as source from 'zip://https://raw.githubusercontent.com/verazuo/jailbreak_llms/refs/heads/main/data/forbidden_question/forbidden_question_set_with_prompts.csv.zip/forbidden_question_set_with_prompts.csv'
    union
    select instruction prompt, False harmful, 'tatsu-lab-alpaca' as source from 'hf://datasets/tatsu-lab/alpaca/**/*.parquet'
    union
    select prompt, True harmful, 'babelscape_alert_adv' as source from 'hf://datasets/Babelscape/ALERT/alert_adversarial.jsonl'
    union
    select prompt, True harmful, 'babelscape_alert' as source from 'hf://datasets/Babelscape/ALERT/alert.jsonl'
    union
    select turns[1], True as harmful, 'sorrybench' as source from 'hf://datasets/sorry-bench/sorry-bench-202406@~parquet/default/train/*.parquet'
    union
    select goal as prompt, True as harmful, 'steering-toxic' as source from 'hf://datasets/Undi95/orthogonal-activation-steering-TOXIC/test.csv'
    union
    select goal as prompt, True as harmful, 'advbench' as source from 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/refs/heads/main/data/advbench/harmful_behaviors.csv'
    union
    select user as prompt, True as harmful, 'many-shot-jailbreaking' as source
    from 'https://raw.githubusercontent.com/KutalVolkan/many-shot-jailbreaking-dataset/refs/heads/main/examples.json'
    where category like 'Harmful%'
    union
    select text as prompt, True as harmful, 'GPTFuzz' as source
    from 'https://github.com/sherdencooper/GPTFuzz/raw/refs/heads/master/datasets/questions/question_list.csv'
)
where not regexp_matches(prompt, '\{[A-Z]+\}|\[[A-Z]+\]', 's')
;

copy dataset to 'dataset.parquet' (
    PARQUET_VERSION v2,
    COMPRESSION zstd,
    COMPRESSION_LEVEL 15
);

from dataset select "source", harmful, count(*) count  group by "source",harmful order by count desc;
"""

import duckdb
print("Downloading datasets... (takes ~30 sec)")
result = (duckdb.sql(SQL).fetchall())

expected = {(src, harm): cnt for src, harm, cnt in EXPECTED_SIZES}
actual = {(src, harm): cnt for src, harm, cnt in result}

print("\nRow counts:")
print(f"{'Source':30}{'Safe?':>5}{'Expected':>10} {'Actual':>10}")
print("-"*70)
for (src, harm) in sorted(set(expected.keys()) | set(actual.keys())):
    expected_count = expected.get((src, harm), 0)
    actual_count = actual.get((src, harm), 0)
    print(f"{src:30}{harm:5}{expected_count:10} {actual_count:10} {'✅ OK' if expected_count == actual_count else '❌ MISMATCH'}")