# Laguna S 2.1 — M5 Max first pass, 2026-08-09

First bounded return of the `BENCHMARK.md` protocol, run against runbook commit
`4323dde9a344be82c29c8ba2da62d188bf65d5a6`.

Read [`SUMMARY.md`](SUMMARY.md) first — it is at revision 2, which corrects the
latency arithmetic, adds the omitted secondary metrics, restates the promotion
scope per quantisation, and records an unresolved power-state question.

In short: the R1 resident route is a parity-clean +4.31 % generation win on
Q2/Q3 and the only experiment clearing the promotion bar, but promotion is
provisional pending an Energy-Mode-controlled rerun of the R1 ABBA matrix
(`SUMMARY.md` §0). The DFlash matrix is invalid because speculation accepted
0/18 draft tokens and self-paused in every arm.

## Contents

| Path | Contents |
| --- | --- |
| `SUMMARY.md` | The required report: host, inputs, all tables, decisions, warnings |
| `metadata/` | Host state before/after, benchmark start state, input SHA-256s, immutable revisions, Metal test logs, worktree status after build, redaction record |
| `collected/` | Flattened per-arm metrics (`bench-metrics.tsv`, `gpu-argmax-metrics.tsv`, `dflash-metrics.tsv`) and the warning scan |
| `raw/<matrix>/<arm>/metrics.csv` | The unmodified `ds4-bench` CSV for every arm, warm-ups included |
| `archive.sha256` | SHA-256 of the published raw-results archive |

## The full archive

The complete evidence tree — every `stdout.log`, `stderr.log`, `decoded.txt`,
frontier-logit JSON, thermal, `vm_stat` and memory-pressure snapshot, and
per-arm `output-sha256.txt`, 767 files — is attached to the release:

**https://github.com/michaelasper/ds4/releases/tag/bench-laguna-s2.1-m5-max-2026-08-09**

```
ds4-laguna-m5-max-results-redacted.tar.gz
sha256 e708a660f65a07000a809b24ace75a9666f72a395af56a5990fb3f45ad56cdce
```

## Redactions

`BENCHMARK.md` records `system_profiler SPHardwareDataType` verbatim, which
includes the machine's serial numbers, hardware UUID and provisioning UDID.
Those values, and the hostname, are replaced with `[redacted]` in
`metadata/host-before.txt` and `metadata/host-after.txt` here and in the
published archive. Nothing else was modified — no measurement, timing, hash,
log line, decoded output or logit file.

`metadata/REDACTIONS.txt` lists every substitution and gives the `sed` command
that reproduces the published archive from the original, so the transformation
is auditable. The unredacted archive built by `05-package.sh`
(`01af914fc57b10ffab014173884524263ae5e5e92767fd8c12690ba1572dee3b`) is
retained locally and can be shared privately for a byte-level audit.

## Reproducing

Follow `BENCHMARK.md` from the repository root. The three immutable revisions
are in `metadata/revisions.csv`; the model and prompt hashes this pass consumed
are in `metadata/input-sha256.txt`.

Pending reruns, in priority order:

1. **R1 ABBA matrix** — rerun with the Energy Mode explicitly selected and
   recorded, per `SUMMARY.md` §0, before using this pass as a promotion gate.
2. **DFlash** — investigate why the Q8_0 draft accepts nothing against the
   Q2/Q3 target at `--dflash-draft 3 --dflash-p-min 0` before remeasuring exact
   multirow batching or staged SWA.
3. **Default Q4 prefill** — the two pre-rebase A observations drifted −4.33 %,
   over the 3 % noise gate, so Q4 prefill deltas from this pass are unusable.
4. **Q4 R1 confirmation** — five alternating repetitions to establish the sign
   of the +0.76 % before the Q4_K R1 down kernel could ever default on.
