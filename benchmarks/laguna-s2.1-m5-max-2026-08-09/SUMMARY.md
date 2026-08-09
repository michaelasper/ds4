# Laguna S 2.1 — M5 Max first pass

Bounded first return per `BENCHMARK.md` (runbook commit
`4323dde9a344be82c29c8ba2da62d188bf65d5a6`). Every arm ran from its own
worktree with `cwd` set to that worktree.

**Headline:** the R1 QMV path is a parity-clean win on Q2/Q3 (+4.31 % gen
tokens/s) and is the only experiment that clears the promotion threshold, but
its promotion is **provisional** pending an Energy-Mode-controlled rerun (§0).
The DFlash matrix is **invalid** — speculation accepted 0/18 draft tokens and
self-paused in all six arms — and must be rerun before any conclusion about
exact multirow batching or staged SWA.

Latency deltas in this report use the runbook formula
`(control / new - 1) * 100`; throughput deltas use `(new / control - 1) * 100`.
Revision 2 of this report corrects latency figures that revision 1 computed
with the throughput formula, corrects the sign of the DFlash prefill A-drift,
adds the omitted `gen_first_ms` / `gen_steady_tps` rows, restates the promotion
scope as quant-specific, and records the unresolved power state above.

## 0. Open blocker — Energy Mode at benchmark time is not established

`BENCHMARK.md` requires Low Power Mode to be disabled. The archive does **not**
prove that it was, and the two recordings in `metadata/host-before.txt`
contradict each other:

| Source in `host-before.txt` | Reading |
| --- | --- |
| `system_profiler SPPowerDataType`, AC Power and Battery Power | `High Power Mode: No`, `Low Power Mode: Yes` |
| `pmset -g custom`, AC Power and Battery Power | `powermode 2` |

The first version of this report claimed `powermode 2` meant High Power Mode
and that Low Power Mode was off. **That claim was unfounded** and is withdrawn.
The machine's power preferences store the two settings as separate keys:

```
/Library/Preferences/com.apple.PowerManagement.<hardware-uuid>.plist
  "AC Power" => { "HighPowerMode" => 0, "LowPowerMode" => 2, ... }
  "Battery Power" => { "HighPowerMode" => 0, "LowPowerMode" => 2, ... }
```

So the single `powermode 2` line printed by `pmset -g custom` is the
`LowPowerMode` key, not a High Power indicator; `HighPowerMode` is explicitly
`0`. System Profiler's `High Power Mode: No` is therefore the consistent
reading, and its `Low Power Mode: Yes` is a rendering of `LowPowerMode = 2`
whose semantics (enabled vs automatic) this pass cannot pin down from the
recorded evidence.

The live machine still reports the same values, so nothing changed after the
run, but that does not resolve what the setting *meant* while the arms
executed. Per the review instruction, the **R1 ABBA matrix must be rerun with
the Energy Mode explicitly selected and recorded** before this pass is used as
a promotion gate. Until then, treat the R1 numbers below as strong but
provisional: they are internally consistent (A-arm generation drift +0.43 %,
prefill control +0.06 %) and both A and B arms ran under the identical,
whatever-it-was power state, so the *comparison* is sound even though the
*absolute* throughput may not represent a correctly configured machine.

All other findings in this report are unaffected by the power question, since
every experiment is a within-session A/B or A/B/C comparison.

## 1. Host identity

| Field | Value |
| --- | --- |
| Mac model | MacBook Pro, `Mac17,6` |
| Chip | Apple M5 Max, 18 CPU cores (6 Super + 12 Performance) |
| GPU cores | 40 |
| RAM | 128 GB (137438953472 bytes) |
| macOS | 26.5.1, build 25F80 (Darwin 25.5.0, xnu-12377.121.6~2) |
| Xcode / Clang | Xcode 26.5 (17F42) / Apple clang 21.0.0 (clang-2100.1.1.101) |
| Power | AC power (140 W USB-C adapter). **Energy Mode not established — see §0** |
| Display | Single built-in Color LCD, Liquid Retina XDR, 3456 × 2234, no external display |
| Thermal | Nominal before (`host-before.txt`, `benchmark-start.txt`), nominal after (`host-after.txt`); no thermal or performance warning recorded at any point |

Machine state was recorded at `utc=2026-08-09T06:29:12Z`; the first matrix
started at `utc=2026-08-09T06:30:12Z` with thermal pressure already nominal.

## 2. Input identity

| Input | Value |
| --- | --- |
| Pre-rebase/basic commit | `448d5695d1c86401a4e9447c440feb983b73e6de` |
| Rebased baseline commit | `729e0cedf54dccfef93fac5138b26d1157aea56a` |
| Candidate commit | `de1a14111f9cc93237d0c8d5b6b7217108c2fc86` |
| `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` | `61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903` |
| `laguna-s-2.1-Q4_K_M.gguf` | `e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a` |
| `laguna-s-2.1-DFlash-Q8_0.gguf` | `b8dd55dddf48a32442c100493b4bdd7e2b2ec25bafcef46d1c977405abdcebf7` |
| `tests/long_context_story_prompt.txt` | `29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666` |
| `speed-bench/promessi_sposi.txt` | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| `promessi-first-600-lines.txt` | `a50224adb48545ded1a76ab176543812a1afb5e76f413dbd7fe8cf256cb9c1c2` |

All three worktree HEADs were verified against the immutable SHAs
(`metadata/revisions.csv`); no branch name or newer tip was substituted.

Metal gates before any timing: `ds4_test --metal-kernels` ended `metal-kernels:
OK`, and `tests/test_glm_q23_metal` reported `Q2/Q3 Metal A/B bit-exact` for
rows 1 and 4. Both exited zero.

## 3. Default three-way comparison

Two observations per revision (mirrored A-B-C-C-B-A), median reported. All
twelve `decoded.txt` and all twelve `frontier_008192.logits.json` files are
byte-identical within each quant — verified independently of the run scripts.

### Q2/Q3 (`laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf`)

| Revision | prefill_tps obs | prefill_tps median | gen_tps obs | gen_tps median | gen_first_ms median | gen_steady_tps median | kvcache_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-rebase (A) | 481.09, 476.36 | 478.73 | 58.26, 59.33 | 58.80 | 24.941 | 60.14 | 0 |
| rebased baseline (B) | 445.30, 478.48 | 461.89 | 59.98, 59.65 | 59.82 | 22.598 | 61.16 | 0 |
| candidate (C) | 468.00, 472.34 | 470.17 | 59.10, 60.05 | 59.58 | 22.273 | 60.89 | 0 |

| Delta | prefill_tps | gen_tps | gen_first_ms | gen_steady_tps |
| --- | --- | --- | --- | --- |
| pre-rebase → rebased baseline | −3.52 % (unreliable, see below) | +1.73 % | +10.37 % faster | +1.69 % |
| rebased baseline → candidate | +1.79 % (unreliable) | −0.40 % | +1.46 % faster | −0.43 % |
| pre-rebase → candidate | −1.79 % (unreliable) | +1.33 % | +11.98 % faster | +1.25 % |

The Q2/Q3 prefill column must not be used: the baseline arm's two observations
are 445.30 and 478.48 t/s, a 7.45 % within-arm spread that is larger than every
delta in the column. The generation columns are stable and usable.

### Q4 (`laguna-s-2.1-Q4_K_M.gguf`)

| Revision | prefill_tps obs | prefill_tps median | gen_tps obs | gen_tps median | gen_first_ms median | gen_steady_tps median | kvcache_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-rebase (A) | 504.63, 482.78 | 493.71 | 59.25, 59.02 | 59.14 | 21.923 | 60.45 | 0 |
| rebased baseline (B) | 504.18, 486.52 | 495.35 | 59.12, 59.23 | 59.18 | 22.359 | 60.48 | 0 |
| candidate (C) | 492.86, 488.17 | 490.52 | 59.20, 59.29 | 59.25 | 21.340 | 60.50 | 0 |

| Delta | prefill_tps | gen_tps | gen_first_ms | gen_steady_tps |
| --- | --- | --- | --- | --- |
| pre-rebase → rebased baseline | +0.33 % (gate failed, see below) | +0.07 % | 1.95 % slower | +0.05 % |
| rebased baseline → candidate | −0.98 % (gate failed) | +0.12 % | +4.77 % faster | +0.03 % |
| pre-rebase → candidate | −0.65 % (gate failed) | +0.19 % | +2.73 % faster | +0.08 % |

**Q4 prefill fails the noise gate.** The two pre-rebase A observations drift by
−4.33 % (504.63 → 482.78 t/s), over the 3 % limit, so no Q4 prefill delta from
this matrix is usable; the matrix should be rerun if Q4 prefill matters. Q4
generation drift is −0.39 % and passes, but every Q4 generation delta is below
0.2 % — that is, integrating `main` and the optimisation series are both
neutral for Q4 generation at this context.

`kvcache_bytes` is 0 in every arm. That is structural, not a measurement
failure: `ds4-bench` emits the session snapshot length in that column, and a
single 8192 frontier with no snapshot reuse has none. The gate "kvcache_bytes
agrees between paired arms" is therefore trivially satisfied and carries no
information in this pass.

## 4. Isolated experiments

| Experiment | metric | A obs | A median | B obs | B median | delta | parity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 QMV Q2/Q3 | gen_tps | 60.22, 60.48 | 60.35 | 62.93, 62.97 | 62.95 | **+4.31 %** | byte-exact |
| R1 QMV Q2/Q3 | gen_steady_tps | 61.58, 61.89 | 61.74 | 64.38, 64.48 | 64.43 | **+4.37 %** | byte-exact |
| R1 QMV Q2/Q3 | prefill_tps (control) | 488.58, 489.39 | 488.99 | 489.82, 488.76 | 489.29 | +0.06 % | byte-exact |
| R1 QMV Q2/Q3 | gen_first_ms | 23.101, 22.350 | 22.726 | 20.862, 20.977 | 20.919 | **+8.63 % faster** | byte-exact |
| R1 QMV Q4 | gen_tps | 59.45, 58.95 | 59.20 | 59.79, 59.51 | 59.65 | +0.76 % | byte-exact |
| R1 QMV Q4 | gen_steady_tps | 60.79, 60.27 | 60.53 | 61.21, 60.81 | 61.01 | +0.79 % | byte-exact |
| R1 QMV Q4 | gen_first_ms | 21.833, 22.814 | 22.323 | 22.700, 21.581 | 22.140 | +0.83 % faster | byte-exact |
| R1 QMV Q4 | prefill_tps (control) | 494.73, 496.25 | 495.49 | 494.87, 494.64 | 494.76 | −0.15 % | byte-exact |
| Paired prefill Q2/Q3 | prefill_tps (primary) | 490.24, 487.78 | 489.01 | 489.07, 493.31 | 491.19 | +0.45 % | byte-exact |
| Paired prefill Q2/Q3 | gen_tps (must not regress) | 59.95, 60.23 | 60.09 | 60.11, 60.23 | 60.17 | +0.13 % | byte-exact |
| Paired prefill Q2/Q3 | gen_steady_tps | 61.29, 61.65 | 61.47 | 61.48, 61.60 | 61.54 | +0.11 % | byte-exact |
| Paired prefill Q2/Q3 | gen_first_ms | 22.065, 22.289 | 22.177 | 21.966, 21.291 | 21.629 | +2.54 % faster | byte-exact |
| GPU argmax Q2/Q3 (`ds4`) | generation t/s | 59.75, 59.64 | 59.70 | 59.18, 59.39 | 59.29 | −0.69 % | byte-exact |
| GPU argmax Q2/Q3 (`ds4`) | prefill t/s (control) | 429.44, 433.68 | 431.56 | 430.64, 433.18 | 431.91 | +0.08 % | byte-exact |

R1 correctness: the decode-only path was proven by the dedicated 8192 → 8193
one-token probe with `--gen-tokens 0`. `frontier_008193.logits.json` is
byte-identical between the clean arm and `DS4_METAL_GLM_QMV_R1=1` for **both**
resident quants (`r1-probe-q23`, `r1-probe-q4`), and the measured arms'
`decoded.txt` and 8192 frontier JSON also match `01-off-A`. The 8192 frontier
JSON alone is not treated as the R1 result.

GPU argmax banner gate: `ds4: Laguna GPU argmax enabled (single-dispatch)`
appears in `02-on-B` and `03-on-B` and in neither A log. All four
`generated.txt` files are byte-identical.

A-drift (first vs last A observation), 3 % gate:

| Matrix | prefill drift | gen drift | verdict |
| --- | --- | --- | --- |
| default-q23 | −0.98 % | +1.84 % | pass |
| default-q4 | **−4.33 %** | −0.39 % | **prefill fails** |
| r1-q23 | +0.17 % | +0.43 % | pass |
| r1-q4 | +0.31 % | −0.84 % | pass |
| paired-prefill | −0.50 % | +0.47 % | pass |
| gpu-argmax | +0.99 % | −0.18 % | pass |
| dflash | +0.50 % | −0.03 % | pass (matrix invalid for other reasons) |

## 5. DFlash — matrix rejected

| Arm | prefill t/s | gen t/s | cycles | drafted | verified | accepted | accepted/verified | pipeline median ms | pipeline p95 ms | generated SHA-256 | speculation active throughout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A `01` legacy multirow | 434.81 | 57.55 | 6 | 18 | 18 | 0 | 0.000 | 40.950 | 41.202 | `9f165f67…` | **no — paused** |
| A `06` legacy multirow | 436.99 | 57.53 | 6 | 18 | 18 | 0 | 0.000 | 40.842 | 41.064 | `9f165f67…` | **no — paused** |
| B `02` default | 436.81 | 57.84 | 6 | 18 | 18 | 0 | 0.000 | 38.420 | 38.615 | `9f165f67…` | **no — paused** |
| B `05` default | 436.68 | 57.97 | 6 | 18 | 18 | 0 | 0.000 | 38.433 | 38.569 | `9f165f67…` | **no — paused** |
| C `03` staged SWA | 437.01 | 57.58 | 6 | 18 | 18 | 0 | 0.000 | 40.162 | 40.352 | `9f165f67…` | **no — paused** |
| C `04` staged SWA | 436.43 | 57.49 | 6 | 18 | 18 | 0 | 0.000 | 40.194 | 40.357 | `9f165f67…` | **no — paused** |

All six `generated.txt` files are byte-identical
(`9f165f67fb05246102c73c2b942992137fa37e3be3b5a178665ebb00887c04da`), and every
log contains multiple `DFlash cycle` lines, so the output-parity and cycle-count
gates pass. Every other DFlash gate fails:

- **Zero acceptance.** Every cycle in every arm is `drafted=3 verified=3
  accepted=0`. The DFlash summary line reads `6 cycles, 0/18 draft tokens
  accepted (0.0%), 0 low-confidence skipped`.
- **Speculation self-paused in all six arms**, e.g. `ds4: DFlash paused for this
  turn (38.52 ms/token speculative vs 16.91 ms normal decode)` after cycle 5 of
  6. `BENCHMARK.md` requires the matrix be rejected if any log reports DFlash
  being paused.

Consequently 250 of the 256 generated tokens came from ordinary decode, and the
A/B/C timings measure a speculation path that was switched off partway through.
The indicative numbers, recorded but **not** to be used for a decision:

| Comparison | pipeline latency | generation throughput |
| --- | --- | --- |
| A → B (exact Q2/Q3 multirow batching) | +6.43 % faster (40.896 → 38.427 ms) | +0.63 % (57.54 → 57.91 t/s) |
| B → C (staged SWA + require mode) | −4.36 % slower (38.427 → 40.178 ms) | −0.64 % (57.91 → 57.54 t/s) |

Staged SWA did take effect — require mode never printed `required Metal Laguna
staged SWA pipeline unavailable`, and C is measurably and reproducibly ~1.75 ms
slower per cycle than B — so this is a genuine (negative) reading of the staged
path, not a silent fallback.

## 6. Decisions

| Opt-in path | Decision | Rationale |
| --- | --- | --- |
| R1 resident route, **Q2/Q3 kernels only** | **promote** (provisional on §0 rerun) | +4.31 % gen, +4.37 % steady gen, +8.63 % faster `gen_first_ms`, prefill control +0.06 %, clearing the 1.5 % bar with no >1 % regression, and byte-exact parity including the dedicated 8193 decode-only probe. |
| R1 resident route, **Q4 kernel** | keep opt-in | +0.76 % gen / +0.79 % steady is below the 1 % noise floor. Q4 is neutral, not harmed, but the evidence does not support making it the default; five alternating repetitions are required first. |
| `DS4_METAL_GLM_QMV_R1` as a single boolean | **must be split before promotion** | See the scope note below. |
| `DS4_LAGUNA_PREFILL_QK_NORM_ROPE_PAIRED` | keep opt-in | +0.45 % on the primary `prefill_tps` is under the 1 % noise floor with no generation regression; needs five alternating repetitions at 2K/16K/32K/65K before a promotion decision. |
| `DS4_METAL_LAGUNA_GPU_ARGMAX` | keep opt-in | Parity and the enable banner are clean, but generation is −0.69 % — within noise and in the wrong direction, so there is no measured benefit at 8K/256 tokens to promote on. |
| Exact Q2/Q3 multirow verifier batching | **rerun** | The apparent +6.43 % pipeline-latency win comes from a matrix where speculation accepted 0/18 tokens and paused; the measurement is not of a working verifier. |
| `DS4_METAL_LAGUNA_STAGED_SWA` | **rerun** (currently failing) | Staged SWA is 4.36 % slower per cycle and 0.64 % lower end-to-end, failing both required criteria — but on an invalid matrix, so it must be remeasured with speculation actually accepting tokens before being rejected outright. |


### Promotion scope — the flag cannot be promoted as-is

`DS4_METAL_GLM_QMV_R1` is one process-wide boolean. In `ds4_metal.m` a single
`glm_qmv_r1` value selects the R1 kernels for all three resident quantisations
at once:

| Block type | Kernels switched by the same boolean |
| --- | --- |
| Q2_K | `kernel_glm_q2_K_pair_swiglu_r1_f32`, `kernel_glm_q2_K_down_r1_f32` |
| Q3_K | `kernel_glm_q3_K_pair_swiglu_r1_f32`, `kernel_glm_q3_K_down_r1_f32` |
| Q4_K | `kernel_glm_q4_K_down_r1_simd_f32` |

Turning the flag on by default would therefore also move Q4_K's
down-projection onto the R1 route, which this pass measures at +0.76 % — inside
the noise floor and below the promotion bar. "Promote globally" and "keep Q4 on
the ordinary route" cannot both hold for the current implementation, so
promotion requires a **quant-specific split** first. Two workable shapes:

1. Make the R1 route the default per block type: derive `glm_qmv_r1` from the
   tensor's quantisation (on for Q2_K and Q3_K, off for Q4_K) and keep
   `DS4_METAL_GLM_QMV_R1` as an override that forces it on everywhere for
   continued Q4 measurement.
2. Split the environment variable into two independent switches — one covering
   the Q2_K/Q3_K pair-SwiGLU and down kernels, one covering the Q4_K down
   kernel — and default only the former to on.

Either way, the Q4_K R1 down kernel stays opt-in until five alternating
repetitions confirm the direction of its +0.76 %. Note also that Q4_K exercises
only the down kernel — there are no Q4 pair-SwiGLU R1 kernels — which is a
plausible mechanical reason the Q4 effect is so much smaller than Q2/Q3's, not
merely a noise artefact.

## 7. Warnings, fallbacks, thermal excursions, failed parity checks, reruns

1. **DFlash speculation paused in all nine DFlash runs** (six measured, three
   warm-ups) with 0/18 draft tokens accepted. Matrix rejected; needs
   investigation of the draft/target pairing before a rerun. This is the one
   finding that blocks a conclusion.
2. **Q4 prefill A-drift −4.33 %**, over the 3 % gate. Q4 prefill deltas are
   discarded; the Q4 matrix should be rerun after a cool-down if Q4 prefill is
   of interest.
3. **Q2/Q3 baseline prefill within-arm spread 7.45 %** (445.30 vs 478.48 t/s).
   Passes the letter of the A-drift gate — which only compares the two A
   observations — but makes all Q2/Q3 prefill deltas unusable.
4. `kvcache_bytes` is 0 in all arms for the structural reason given in §3, not
   because of a failure.
5. No thermal or performance warning was recorded at any point; no swap or
   memory-pressure event occurred in one arm but not its peer.
6. No `failed`, `unavailable`, `fallback`, `mismatch`, or invalid-token message
   appears in any measured stderr. The only matches from the warning scan
   (`collected/warnings.txt`) are the nine DFlash pause lines in item 1 and
   `mismatch=0` exactness lines in the Metal kernel test log.
7. **Energy Mode at benchmark time is unresolved** (§0). The R1 ABBA matrix
   must be rerun with the setting explicitly selected and recorded before this
   pass is used as a promotion gate.
8. Revision 1 of this report contained arithmetic and scope errors, all
   corrected here: latency deltas computed with the throughput formula
   (`gen_first_ms` in both default matrices and in R1 Q2/Q3), the DFlash prefill
   A-drift sign (+0.50 %, not −0.50 %), omitted `gen_first_ms` and
   `gen_steady_tps` rows for R1 Q4 and paired prefill, an unfounded "High Power
   Mode" claim, and a promotion decision that ignored the flag being a single
   quant-agnostic boolean.
9. No matrix was rerun in this pass. Items 1, 2 and 7 are the pending reruns.

## Evidence layout

- `metadata/` — host state before/after, benchmark start state, input SHA-256s,
  revisions, Metal test logs, worktree status after build.
- `<matrix>/<arm>/` — `command.txt`, `env-overrides.txt`, `revision.txt`,
  `start-utc.txt`, `stdout.log`, `stderr.log`, `metrics.csv`, `decoded.txt`,
  `logits/`, thermal/vm snapshots, `output-sha256.txt`.
- `collected/` — `bench-metrics.tsv`, `gpu-argmax-metrics.tsv`,
  `dflash-metrics.tsv`, `warnings.txt`, derived from the raw logs by
  `06-collect-metrics.sh`.
- `MANIFEST.txt` — every file in this tree.
