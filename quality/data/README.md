# Laguna quality fixture

The release fixture is `laguna-openrouter-100`: 100 deterministic Laguna S2.1
continuations collected through OpenRouter model `poolside/laguna-s-2.1`.

It contains:

- `prompts/case_*.txt`: exact user prompts;
- `continuations/case_*.txt`: hosted-model continuations;
- `responses/case_*.json`: raw hosted responses retained for provenance;
- `manifest.tsv`: relative prompt and continuation paths consumed by
  `score_official`.

Poolside does not expose output-token logprobs for this model, so the manifest
intentionally omits the optional raw-response column. The scorer reports local
target-token NLL, first-token agreement, and greedy-prefix agreement; API
top-logprob metrics are unavailable. This is the only tracked quality fixture.
