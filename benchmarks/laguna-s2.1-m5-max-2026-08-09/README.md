# Laguna S 2.1 — M5 Max first pass, 2026-08-09

First bounded return of the `BENCHMARK.md` protocol, run against runbook commit
`4323dde9a344be82c29c8ba2da62d188bf65d5a6`.

Read [`SUMMARY.md`](SUMMARY.md) first. In short: `DS4_METAL_GLM_QMV_R1` is a
parity-clean +4.31 % generation win on Q2/Q3 and is the only experiment that
clears the promotion bar; the DFlash matrix is invalid because speculation
accepted 0/18 draft tokens and self-paused in every arm.

## Contents

| Path | Contents |
| --- | --- |
| `SUMMARY.md` | The required report: host, inputs, all tables, decisions, warnings |
| `metadata/` | Host state before/after, benchmark start state, input SHA-256s, immutable revisions, Metal test logs, worktree status after build |
| `collected/` | Flattened per-arm metrics (`bench-metrics.tsv`, `gpu-argmax-metrics.tsv`, `dflash-metrics.tsv`) and the warning scan |
| `raw/<matrix>/<arm>/metrics.csv` | The unmodified `ds4-bench` CSV for every arm, warm-ups included |
| `archive.sha256` | SHA-256 of the full raw-results archive |

## The full archive

The complete evidence tree — every `stdout.log`, `stderr.log`, `decoded.txt`,
frontier-logit JSON, thermal and `vm_stat` snapshot, and per-arm
`output-sha256.txt` — is 23 MB compressed and is kept outside git:

```
ds4-laguna-m5-max-results.tar.gz
sha256 01af914fc57b10ffab014173884524263ae5e5e92767fd8c12690ba1572dee3b
```

Ask for it if you need to re-check a parity claim; everything in `SUMMARY.md`
is derived from it.

## Redactions

`metadata/host-before.txt` and `metadata/host-after.txt` have the machine
serial numbers, hardware UUID, provisioning UDID, and hostname replaced with
`[redacted]`. Nothing else in these files was modified.

## Reproducing

Follow `BENCHMARK.md` from the repository root. The three immutable revisions
are recorded in `metadata/revisions.csv`; the model and prompt hashes that this
pass consumed are in `metadata/input-sha256.txt`.

Two matrices need a rerun before their questions are settled:

1. **DFlash** — investigate why the Q8_0 draft model accepts nothing against
   the Q2/Q3 target at `--dflash-draft 3 --dflash-p-min 0` before remeasuring
   exact multirow batching or staged SWA.
2. **Default Q4 prefill** — the two pre-rebase A observations drifted −4.33 %,
   over the 3 % noise gate, so Q4 prefill deltas from this pass are unusable.
