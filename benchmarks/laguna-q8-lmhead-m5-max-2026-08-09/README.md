# Laguna Q8 lm-head screen — M5 Max bounded return, 2026-08-09

Bounded first pass per `BENCHMARK.md` @ `6ef3f02e9311215114ecec33cd705556c267d5ae`
(`sha256 30168f20ff355c2de315060d8558c64402d14c7c9649ca2391a390c8636496b2`,
verified before the run), candidate `202a46b5158bff38ba36e066d71bc5b242b53a4a`.

Run from a fresh `DS4_BENCH_ROOT`, separate from the earlier R1 evidence, which
it does not touch or amend.

## Result: reject the screen as a default

Correctness is exact; throughput is not there.

| Gate | Value | Required | |
| --- | --- | --- | --- |
| winner ID / float32 bits vs stock | `329` / `0x41aac44b`, both exact | exact | pass |
| generated output parity | byte-identical across the screen matrix | exact | pass |
| `coarse_nonfinite` on every trace row | 0 | 0 | pass |
| `screen_generation_ratio` | 0.986483 | ≥ 1.015 | **fail** |
| `screen_prefill_ratio` | 0.999364 | ≥ 0.99 | pass |
| `net_ratio` | 0.967435 | ≥ 0.99 | **fail** |
| `net_prefill_ratio` | 0.989130 | ≥ 0.99 | **fail** |

Semantic ledger is `status=FAIL`, so `promote` is not available.

The interesting tension: the trace says the screen touches at most 1 353 600
exact row-blocks against a 4 533 550 break-even — roughly 42 % fewer weight
bytes than the raw head — yet measured generation is 1.35 % *slower*. Reduced
traffic is not turning into reduced time, so the cost is dispatch/gather
structure, not memory bandwidth. That is the thing to explain before this path
is re-proposed.

Separately, **safe Metal math changes the generated text** (712 → 717 bytes,
each pair internally byte-identical), which trips the safe-tax matrix's output
parity requirement, and costs 1.93 % generation / 1.02 % prefill.

## Two things needing review sign-off

1. **Power-gate deviation.** The runbook's `SPPowerDataType` High/Low Power
   check cannot pass on macOS 26.5.1 in any Energy Mode — this build encodes
   High Power as `LowPowerMode`/`powermode = 2` and leaves the legacy
   `HighPowerMode` key at 0. The gate was overridden with a documented rule and
   the original verdict is recorded alongside every capture. See
   `metadata/POWER-GATE-DEVIATION.txt` and `SUMMARY.md` §0.
2. **Runbook internal path discrepancy.** `clean-q23-*.csv` is written into
   `default-q23/` by the generator but required at the results root by the
   packager. See `metadata/RUNBOOK-PATH-DISCREPANCY.txt`.

## Contents

| Path | Contents |
| --- | --- |
| `SUMMARY.md` | Full report: host, inputs, clean Q2/Q3, lm-head A–E, decisions, warnings |
| `clean-q23-metrics.csv`, `clean-q23-deltas.csv` | Runbook-generated three-way metrics and deltas |
| `metadata/` | Revisions, input hashes, semantic ledger, net-gate CSV + status, power/host state, deviation and redaction records |
| `lmhead/` | Trace CSV + parser summary, winner-parity CSV + status |
| `archive.sha256` | SHA-256 of the full evidence archive |

## Full archive

776 members, every per-arm command, environment, revision, stdout/stderr,
generated output, decoded text, 8192 frontier logits, token-count record,
output hash, and battery/thermal/power-profile/vm-stat/swapusage/
memory-pressure snapshot before and after:

**https://github.com/michaelasper/ds4/releases/tag/bench-laguna-q8-lmhead-m5-max-2026-08-09**

```
ds4-laguna-m5-max-results.tar.gz
sha256 88a7f027098827f7864e99576ff7f20cc813725bd43af5a45e9163a5d9cfe0a7
```

Packed with `COPYFILE_DISABLE=1`; no AppleDouble `._*` sidecars. Redactions
beyond the runbook's own are listed in `metadata/REDACTIONS.txt`.
