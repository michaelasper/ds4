# Laguna continuation quality testing

This directory compares local Laguna S2.1 GGUF variants with deterministic
hosted-model continuations. The primary metric is target-token negative log
likelihood: collect one reference continuation, then measure how much
probability each local GGUF assigns to those exact tokens. This is more useful
than judging a quantization from one sampled answer.

## Tracked fixture

`data/laguna-openrouter-100` contains 100 continuations collected from
OpenRouter model `poolside/laguna-s-2.1`.

The Poolside endpoint does not expose output-token logprobs. The fixture
therefore supports target-token NLL, first-token agreement, and greedy-prefix
comparison. API logprob-delta and top-N agreement fields are zero by design.
Raw responses are retained for provenance but omitted from `manifest.tsv`.

## Refreshing the fixture

Refreshing a committed fixture is an intentional data change. Use a new output
directory, review the prompts and responses, and never overwrite the tracked
set merely to make a candidate score better.

```zsh
export OPENROUTER_API_KEY=...

python3 gguf-tools/quality-testing/collect_official.py \
  --model poolside/laguna-s-2.1 \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out /tmp/laguna-openrouter-candidate \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 0 \
  --token-limit-field max_tokens \
  --thinking omit \
  --reasoning-effort none
```

The collector writes exact prompts, continuations, raw JSON responses, and a
relative `manifest.tsv`.

## Building the scorer

The supported scorer uses the project runtime and Apple Metal:

```zsh
make gguf-tools/quality-testing/score_official
```

The optional `score_llama` program is an external llama.cpp control and is not
part of the Laguna runtime build.

## Scoring GGUF variants

Score the old and new GGUF against the same manifest and context:

```zsh
gguf-tools/quality-testing/score_official \
  /path/to/OLD.gguf \
  gguf-tools/quality-testing/data/laguna-openrouter-100/manifest.tsv \
  /tmp/old.tsv 4096 --quality

gguf-tools/quality-testing/score_official \
  /path/to/NEW.gguf \
  gguf-tools/quality-testing/data/laguna-openrouter-100/manifest.tsv \
  /tmp/new.tsv 4096 --quality

python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

`--quality` selects the runtime's exact/reference choices where an explicitly
faster numerical path exists. For release evidence, record the model hashes,
commit, command, hardware, raw TSVs, and comparator output.

Important output fields:

- `avg_nll`: average negative log likelihood; lower is better;
- `delta_new_minus_old`: negative favors the new GGUF;
- `case_wins_new_old_ties`: per-prompt NLL comparison;
- `first_token_matches`: local greedy first-token agreement;
- `avg_greedy_lcp`: average greedy longest common prefix.

Do not compare manifests from another model family or checkpoint, and do not
interpret unavailable API-logprob fields as measured zeros.
