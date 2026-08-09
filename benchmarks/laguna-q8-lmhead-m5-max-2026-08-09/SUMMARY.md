# Laguna S 2.1 — certified Q8 lm-head screen, M5 Max bounded return

Bounded first pass per `BENCHMARK.md` @
`6ef3f02e9311215114ecec33cd705556c267d5ae`
(`sha256 30168f20ff355c2de315060d8558c64402d14c7c9649ca2391a390c8636496b2`,
verified before the run). Only the current bounded first pass was executed; no
optional or historical section was run.

**Decision: `reject` the Q8 lm-head screen as a default. Correctness is
flawless; throughput is not.** The screen's exact top-1 winner matches the
stock full-logit path bit-for-bit and its generated output is byte-identical,
but generation is **1.35 % slower**, missing the required +1.5 %, and both net
ratios fall below 0.99. The semantic ledger is `status=FAIL`, so `promote` is
not available under the runbook's own rule.

A second, independent finding: **safe Metal math changes the generated text**.
The safe-mode-tax matrix requires all four outputs to match; the two fast-math
arms produce a 712-byte completion and the two safe-math arms a 717-byte one,
each pair internally identical. That is a deterministic numerics difference,
not flakiness, and it is recorded as a semantic reject.

## 0. Documented deviation — the power gate

The runbook's `verify_power_and_thermals` requires
`system_profiler SPPowerDataType` to report, under AC Power,
`High Power Mode: Yes` and `Low Power Mode: No`. **That output is unobtainable
on this host** even with Energy Mode explicitly set to High Power:

| Source | Reading |
| --- | --- |
| System Settings › Battery › Energy Mode | On battery: **High Power**, On power adapter: **High Power** |
| `pmset -g custom` | AC Power `powermode 2`, Battery Power `powermode 2` |
| `com.apple.PowerManagement.<uuid>.plist` | AC + Battery: `HighPowerMode = 0`, `LowPowerMode = 2` |
| `system_profiler SPPowerDataType` | AC + Battery: `High Power Mode: No`, `Low Power Mode: Yes` |

macOS 26.5.1 (25F80) encodes Energy Mode = High Power as `LowPowerMode`/
`powermode = 2` and leaves the legacy `HighPowerMode` key at `0`;
`system_profiler` renders any non-zero `LowPowerMode` as
`Low Power Mode: Yes`. The runbook gate therefore tests a key this build no
longer populates, and cannot pass on this machine in any Energy Mode.

The gate was overridden with a documented rule that keeps the runbook's
thermal test verbatim and requires: AC is the current source, AC
`powermode == 2`, plist AC `HighPowerMode == 0` and `LowPowerMode == 2`, and
nominal thermals. Every arm still captures the **raw, unmodified**
`SPPowerDataType` JSON, and each `power-*.txt` records the original gate's
verdict as `runbook_spprofiler_gate=fail` next to `power_gate=pass`, so an
auditor can re-derive the state and re-apply the original rule.

Full record and the override source: `metadata/POWER-GATE-DEVIATION.txt` and
`lib/power-gate-override.sh` in the harness. **This deviation needs explicit
review sign-off**, and the runbook's gate should be fixed for macOS 26.

This also partly reverses a claim in the earlier R1 report: `powermode 2` on
this machine *does* correspond to High Power, so that report's §0 withdrawal
was itself over-corrected. The plist has been unmodified since Aug 7, i.e. the
machine was already in High Power during the R1 run.

## 1. Host identity

| Field | Value |
| --- | --- |
| Mac model | MacBook Pro, `Mac17,6` |
| Chip | Apple M5 Max, 18 CPU cores (6 P-class + 12) |
| GPU cores | 40 |
| RAM | 128 GB (137438953472 bytes) |
| macOS | 26.5.1, build 25F80 (Darwin 25.5.0, xnu-12377.121.6~2) |
| Xcode / Clang | Xcode 26.5 (17F42) / Apple clang 21.0.0 (clang-2100.1.1.101) |
| Power | AC (140 W adapter), Energy Mode High Power on both sources — see §0 |
| Display | Single built-in Color LCD, 1728 × 1117 @ 120 Hz, main, no external display |
| Thermal | Nominal throughout; no thermal, performance or CPU-power warning recorded at any checkpoint |

Host identity fields are `<redacted>` by the runbook's own
`capture_profiler_json` before any snapshot is retained.

## 2. Input identity

| Input | Value |
| --- | --- |
| Runbook commit | `6ef3f02e9311215114ecec33cd705556c267d5ae` |
| Runbook `BENCHMARK.md` | `30168f20ff355c2de315060d8558c64402d14c7c9649ca2391a390c8636496b2` |
| Pre-rebase/basic | `448d5695d1c86401a4e9447c440feb983b73e6de` |
| Rebased baseline | `729e0cedf54dccfef93fac5138b26d1157aea56a` |
| Candidate | `202a46b5158bff38ba36e066d71bc5b242b53a4a` |
| `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` | `61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903` |
| `tests/long_context_story_prompt.txt` | `29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666` |
| `speed-bench/promessi_sposi.txt` | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| `promessi-first-600-lines.txt` | `a50224adb48545ded1a76ab176543812a1afb5e76f413dbd7fe8cf256cb9c1c2` |

Worktree HEADs verified against the immutable SHAs (`metadata/revisions.csv`).
Q4 and DFlash models were not required and were not touched.

Pre-timing gates: `ds4_test --metal-kernels` ended `metal-kernels: OK`; the
focused screen smoke ended `metal-laguna-q8-lmhead-screen: OK` with
`packed_bytes=173408256`, matching the runbook's expected sidecopy size.

## 3. Clean Q2/Q3 three-way

Mirrored A-B-C-C-B-A, two observations per revision, median reported. All six
`decoded.txt` and all six `frontier_008192.logits.json` are byte-identical.
Within-revision drift on the primary metric (`gen_steady_tps`): pre 0.691 %,
base 0.933 %, candidate 0.033 % — all inside the 3 % gate.

| Revision | prefill_tps obs | median | gen_tps obs | median | gen_steady_tps obs | median | gen_first_ms obs | median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre-rebase/basic | 464.73, 498.36 | 481.545 | 59.48, 59.85 | 59.665 | 60.75, 61.17 | 60.960 | 20.769, 21.708 | 21.2385 |
| rebased baseline | 493.86, 500.98 | 497.420 | 59.81, 60.30 | 60.055 | 61.08, 61.65 | 61.365 | 20.537, 21.960 | 21.2485 |
| candidate | 499.49, 502.85 | 501.170 | 60.19, 60.13 | 60.160 | 61.53, 61.51 | 61.520 | 21.951, 23.680 | 22.8155 |

| Comparison | prefill_tps | gen_tps | gen_steady_tps | gen_first_ms |
| --- | --- | --- | --- | --- |
| pre-rebase → rebased baseline | +3.30 % | +0.65 % | +0.66 % | 0.05 % slower |
| rebased baseline → candidate | +0.75 % | +0.17 % | +0.25 % | **6.87 % slower** |
| pre-rebase → candidate | +4.08 % | +0.83 % | +0.92 % | **6.91 % slower** |

Throughput deltas use `(right/left - 1) * 100`; `gen_first_ms` uses
`(left/right - 1) * 100`, both computed by the runbook's own helper into
`default-q23/clean-q23-deltas.csv`.

Reading: integrating `main` is worth ~3.3 % prefill and ~0.7 % generation; the
optimisation series adds a further ~0.75 % prefill and ~0.25 % generation. All
generation deltas are under 1 %, i.e. at or below the noise floor. The
`gen_first_ms` regression is the one number that stands out, but its two
candidate observations are 21.951 and 23.680 ms — a 7.9 % within-arm spread
that is larger than the delta itself, so first-token latency needs repetitions
before anyone acts on it.

## 4. Candidate lm-head A–E

### A. Safe-mode tax (fast vs safe math, no screen)

| Arm | prefill t/s | generation t/s | output SHA-256 | requested / bytes |
| --- | --- | --- | --- | --- |
| 01-fast-A | 438.21 | 59.26 | `9f165f67…` | 256 / 712 |
| 02-safe-B | 433.87 | 58.19 | `0ea73c4a…` | 256 / 717 |
| 03-safe-B | 433.06 | 58.12 | `0ea73c4a…` | 256 / 717 |
| 04-fast-A | 437.69 | 59.34 | `9f165f67…` | 256 / 712 |

Medians: fast 59.30, safe 58.155 generation; fast 437.95, safe 433.465 prefill.
**Safe-math tax: generation ×0.98069 (−1.93 %), prefill ×0.98976 (−1.02 %).**
A-arm drift 0.135 %.

**Semantic reject:** the runbook requires all four `generated.txt` and
`token-count.txt` to match. They do not — fast math yields 712 bytes, safe math
717. Each pair is internally byte-identical across both repetitions, so this is
a deterministic fast-vs-safe numerics divergence, not nondeterminism. Both
B arms carry `Metal shader library math mode = safe`; both A arms carry
`math_safe=off`; GPU-argmax diagnostic present in all four.

### B. Screen throughput (safe math, screen off vs on)

| Arm | prefill t/s | generation t/s | output SHA-256 | requested / bytes |
| --- | --- | --- | --- | --- |
| 01-safe-off-A | 434.39 | 57.91 | `0ea73c4a…` | 256 / 717 |
| 02-safe-on-B | 432.91 | 57.27 | `0ea73c4a…` | 256 / 717 |
| 03-safe-on-B | 431.93 | 57.31 | `0ea73c4a…` | 256 / 717 |
| 04-safe-off-A | 431.00 | 58.24 | `0ea73c4a…` | 256 / 717 |

All four outputs and token-count records are byte-identical. No trace,
fallback, unavailable or failed diagnostic in any timed arm; both B arms carry
`screen prepared` and `screen enabled (exact top-1 only)`; neither A arm does.
A-arm drift 0.570 %.

Medians: screen-off 58.075, screen-on 57.290 generation; 432.695 vs 432.420
prefill.

| Ratio | Value | Threshold | Verdict |
| --- | --- | --- | --- |
| `safe_tax_ratio` (generation) | 0.980691 | — | reported |
| `safe_prefill_ratio` | 0.989759 | — | reported |
| `screen_generation_ratio` | **0.986483** | ≥ 1.015 | **fail** |
| `screen_prefill_ratio` | 0.999364 | ≥ 0.99 | pass |
| `net_ratio` | **0.967435** | ≥ 0.99 | **fail** |
| `net_prefill_ratio` | **0.989130** | ≥ 0.99 | **fail** |
| safe-tax A drift | 0.00135 | ≤ 0.03 | pass |
| screen A drift | 0.00570 | ≤ 0.03 | pass |

`performance_gate=False` (`metadata/lmhead-net-gate.csv`). The screen costs
1.35 % of generation throughput rather than adding 1.5 %, and prefill is
neutral (−0.06 %).

### C. One-token init / RSS

| Arm | max RSS (bytes) | screen diagnostic |
| --- | --- | --- |
| safe-off | 48 316 514 304 | none (correctly absent) |
| safe-on | 48 316 563 456 | `screen prepared (packed 165.38 MiB, 173408256 bytes, init 7.052 ms)` |

Delta 49 152 bytes of maximum RSS — the 165.38 MiB packed sidecopy does not
show up as extra peak RSS at this measurement point. `sidecopy_init_ms` is
7.052 ms (allocation + sidecopy only). Both arms carry the GPU-argmax and safe
math diagnostics; the screen is absent from the safe-off control.

### D. Screen trace (16 tokens) + trace-off control

Parser `status=PASS`, 16 rows, `screen_calls` 1…16, `coarse_nonfinite=0` on
every real-prompt row. Trace-on and trace-off generated outputs are
byte-identical, and the control emitted no screen-stats line.

| Quantity | min | median | max |
| --- | --- | --- | --- |
| `candidate_rows` | 71 | ~2 573 | 7 542 |
| `exact_row_blocks` | 13 632 | 481 632 | 1 353 600 |
| `candidate_percent` | 0.071 % | 2.56 % | 7.52 % |
| `exact_physical_row_block_percent` | 0.14 % | 5.00 % | 14.05 % |
| `exact_q8_bytes` | 463 488 | 16.4 M | 46.0 M |
| `estimated_weight_bytes` | 173.9 M | 189.8 M | 219.4 M |

Constants: `total_rows=100352`, `blocks_per_row=96`, raw head bytes
`327 548 928`, packed sidecopy `173 408 256`. Break-even is
`exact_row_blocks < 4 533 550`; **no row came close** — the maximum is 1 353 600,
30 % of break-even, and the median estimated weight traffic is 189.8 MB against
327.5 MB raw.

This is the interesting tension in the result: **the traffic model predicts a
comfortable win and the wall clock says the opposite.** Reduced bytes are not
translating into reduced time, so the cost is elsewhere — dispatch overhead,
the candidate-gather step, or the two-pass structure — not memory traffic.
That is where to look before re-proposing this path.

### E. Exact one-token winner-value probe

| Field | Value |
| --- | --- |
| stock rank-0 ID | 329 |
| stock rank-0 printed | 21.3458462 |
| stock rank-0 float32 bits | `0x41aac44b` |
| screen winner ID | 329 |
| screen winner bits | `0x41aac44b` |
| winner ID exact | true |
| winner value bits exact | true |
| generated output byte exact | true |
| `parity_status` | **pass** |

The screen returns the identical winner token and the identical float32 winner
value as the stock full-logit path. Correctness of the certified screen is not
in question.

## 5. Decisions

| Path | Decision | Grounds |
| --- | --- | --- |
| `DS4_METAL_LAGUNA_Q8_LMHEAD_SCREEN` | **reject** as a default | Exact winner ID/bits and byte-exact output, but `screen_generation_ratio=0.9865` against a ≥1.015 requirement, `net_ratio=0.9674` and `net_prefill_ratio=0.9891` against ≥0.99. Semantic ledger `status=FAIL` forbids `promote`. Keep the code opt-in and behind its flag; it is correct, just not faster here. |
| `DS4_METAL_MATH_SAFE` (safe-mode tax) | reported separately, **not promotable as-is** | Costs 1.93 % generation and 1.02 % prefill, and changes generated text (712 → 717 bytes). The output change is the blocking issue, not the tax. |
| Candidate revision `202a46b` vs baseline | no action needed | +0.75 % prefill, +0.25 % generation — inside noise; `gen_first_ms` regression is smaller than its own within-arm spread. |

The screen is worth revisiting only if the gap between its traffic model (§D:
~42 % fewer weight bytes) and its measured throughput (−1.35 %) is explained
and closed. Five alternating repetitions would not rescue a −1.35 % median.

## 6. Warnings, diagnostics, deviations

1. **Power-gate deviation (§0)** — the runbook's `SPPowerDataType` check cannot
   pass on macOS 26.5.1 in any Energy Mode; an overridden, fully documented
   gate was used and the original verdict is recorded alongside. Needs review
   sign-off, and the runbook gate should be fixed.
2. **Semantic ledger `status=FAIL`** with five entries: safe-tax generated
   output and token/output-byte mismatch on `02-safe-B` and `03-safe-B`, plus
   the executable performance-threshold failure.
3. Expected intentional diagnostic in the focused smoke:
   `Laguna Q8 lm-head screen trace allows only one call per active command
   batch`.
4. No unexpected `failed`, `unavailable`, `fallback`, `paused`, `mismatch` or
   invalid-token message in any measured stderr. No thermal, performance or
   CPU-power warning at any checkpoint. No swap or memory-pressure event in one
   arm but not its peer.
5. Every A-arm drift gate passed (clean Q2/Q3 ≤0.93 %, safe-tax 0.14 %, screen
   0.57 %).
6. **An earlier attempt at this return was discarded** before any measurement:
   a harness bug let a failed power gate be reported as a successful build, and
   ~55 arms then ran against binaries that did not exist. Per the runbook that
   tree was abandoned and this return was produced from a fresh
   `DS4_BENCH_ROOT`; the failed attempt is retained outside the archive with a
   written root-cause note and is not part of this evidence.
7. **Excluded historical sections were not run**: R1 one-row-per-SIMD QMV,
   full-ring SWA GQA3, paired prefill Q/K norm and RoPE, direct raw GPU argmax,
   the Q4 quant, and DFlash. No Q4/R1/GQA3/paired-prefill/argmax/DFlash rows
   appear in this return.
8. **Runbook internal path discrepancy.** The clean-Q2/Q3 helper writes
   `clean-q23-metrics.csv` / `clean-q23-deltas.csv` into
   `$DS4_RESULTS/default-q23/`, but the packager's required-artifact list
   expects them at `$DS4_RESULTS/`. The generated files were left untouched and
   byte-identical copies were placed at the required paths; both locations
   appear in `MANIFEST.txt` with matching SHA-256. See
   `metadata/RUNBOOK-PATH-DISCREPANCY.txt`.
9. **Two redactions beyond the runbook's own.** The power-gate override dumps
   the PowerManagement preference file, which the runbook never reads; its
   `BatteryWarn` block is keyed by the battery serial and its filename embeds
   the hardware UUID. Both were replaced with `<redacted>` in the ten affected
   `power-*.txt` files before packaging, and the override now redacts at
   capture time. No measurement, hash, log, output or gate verdict was touched.
   See `metadata/REDACTIONS.txt`.
10. The optional 2K–65K extended sweep was not run; it does not block this
    return.
