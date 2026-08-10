# Laguna S 2.1 M5 Max finalist-confirmation runbook

This is the executable how-to guide for the benchmark that follows the
2026-08-10 discovery screen. It answers one question: which of the three
discovered decode finalists, alone or in the two sensible GQA3×ladder
combinations, should be promoted?

Run the committed driver rather than copying shell fragments from this file:

```zsh
benchmark/run_m5_finalists.zsh all
```

The driver owns the commands, environment, randomisation, validation,
analysis, packaging, and draft-release upload. `BENCHMARK.md`, the driver, and
the analyser are copied into the result archive and hashed before any timing.
Do not modify any of them after starting a result root.

## Remote-agent execution contract

This section is the complete handoff for an agent that receives only
`BENCHMARK.md`. Treat the committed driver and analyser as the executable
implementation of this contract. Do not reconstruct the benchmark from prose,
change selector values, substitute revisions, or invent recovery steps.

The task has three terminal stages:

1. `all` returns zero and creates a verified local package;
2. `upload-draft` returns zero and prints a verified GitHub draft URL;
3. the draft is published only when the operator explicitly requested a public
   release and the downloaded `SUMMARY.md` agrees with the decision ledger.

If the instruction is merely to run the benchmark, stop after stage 2 and
return the draft URL. Never publish merely because a draft exists.

### Non-negotiable operating rules

- Run only on macOS on an Apple M5 Max with exactly 128 GiB RAM. The runner
  fails closed on another host.
- `laguna-s2.1` is the repository's default and authoritative branch. Do not
  switch to, merge, or rebase onto `main`.
- Start from a clean, detached checkout of the current remote
  `laguna-s2.1` tip. The runner records that exact runbook commit; the timed
  and trace binaries still use the separately pinned revisions below.
- Use one fresh `DS4_BENCH_ROOT` for one invocation of `all`. Never resume,
  repair, prune, or overwrite an interrupted root.
- Keep the process in the foreground, or keep polling the same yielded process.
  Long periods without terminal output are expected. Never start a second
  `all` because the first appears quiet.
- Do not edit `BENCHMARK.md`, the runner, the analyser, any status file, or the
  checkout after `prepare` begins.
- Do not run another GPU-, CPU-, memory-, or storage-intensive workload during
  measurement.
- Do not add a power-mode check or operator override. Power, AC/battery,
  thermal, Energy Mode, and display values are informational only.
- Preserve every `SEMANTIC_REJECT` and `NOT_ELIGIBLE` result exactly as
  produced. They are valid outcomes, not invitations to rerun an arm.
- Never reuse the discovery tag or mutate an existing release. A failed draft
  upload requires inspection and a new unused tag.

The only expected reasons to request operator input are: no writable fast
volume with enough model space, missing GitHub authentication when a draft is
requested, or missing authority to publish the verified draft. Do not weaken
the protocol to work around any of them.

If `all` exits nonzero or the controlling process is lost, stop and return the
exact `DS4_BENCH_ROOT`, failing command, exit status, and last phase marker to
the operator. Do not launch a second expensive `all` invocation without
explicit approval.

### End-to-end command sequence

Use a fresh protocol checkout. This avoids relying on remote names or local
branches configured by a previous agent. Run the blocks in order and stop on
the first nonzero command or failed `[[ ... ]]` assertion:

```zsh
command -v git
command -v make
command -v python3
command -v gh
xcode-select -p >/dev/null

export DS4_RUNBOOK_REPO="$(mktemp -d /private/tmp/ds4-finalist-runbook.XXXXXX)"
git clone --branch laguna-s2.1 --single-branch \
  https://github.com/michaelasper/ds4.git "$DS4_RUNBOOK_REPO"
cd "$DS4_RUNBOOK_REPO"

runbook_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote https://github.com/michaelasper/ds4.git \
  refs/heads/laguna-s2.1 | awk '{print $1}')
[[ -n "$remote_sha" && "$runbook_sha" == "$remote_sha" ]]
git switch --detach "$runbook_sha"
[[ -z "$(git status --porcelain=v1)" ]]
```

Set the model directory to an absolute path on a fast volume. The required
model is approximately 45 GiB. If it is not already present, ensure the model
volume has at least 55 GiB free before continuing; the runner downloads it
automatically. The result root contains worktrees, builds, logs, and the final
archive, but only a symlink to the model.

If no suitable model volume is mounted, stop before creating or running the
benchmark root and report the storage requirement. Do not delete unrelated
files or attempt a partial download.

```zsh
export DS4_GGUF_DIR=/absolute/path/to/laguna-models
mkdir -p "$DS4_GGUF_DIR"
df -h "$DS4_GGUF_DIR" /private/tmp

model="$DS4_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf"
if [[ -f "$model" ]]; then
  print -- 'Model present; the runner will verify its pinned SHA-256 once.'
else
  print -- 'The runner will download the approximately 45 GiB model.'
  if ! command -v hf >/dev/null 2>&1; then
    python3 -m pip install --user -U huggingface_hub hf_xet
  fi
fi

export DS4_BENCH_ROOT="$(mktemp -d /private/tmp/ds4-laguna-finalists.XXXXXX)"
export DS4_JOBS=8
export DS4_COOLDOWN_SECONDS=30
```

Run the inexpensive protocol tests before starting the costly model loads:

```zsh
cd "$DS4_RUNBOOK_REPO"
zsh -n benchmark/run_m5_finalists.zsh benchmark/test_run_m5_finalists.zsh
zsh benchmark/test_run_m5_finalists.zsh
python3 -m unittest -v benchmark/test_m5_finalist_analysis.py
gh auth status
```

Now run the official workflow exactly once:

```zsh
cd "$DS4_RUNBOOK_REPO"
benchmark/run_m5_finalists.zsh all
```

The official invocation is intentionally long-running. Progress may be
observed read-only through these files; their absence means that phase has not
completed yet:

```text
$DS4_BENCH_ROOT/results/metadata/prepare-status.txt   PASS
$DS4_BENCH_ROOT/results/metadata/prove-phase.txt     COMPLETE
$DS4_BENCH_ROOT/results/metadata/measure-phase.txt   COMPLETE
$DS4_BENCH_ROOT/results/metadata/analyse-phase.txt   COMPLETE
$DS4_BENCH_ROOT/package-complete.txt                 status=COMPLETE
```

When `all` returns zero, verify and inspect the result before any network
write:

```zsh
sed -n '1,240p' "$DS4_BENCH_ROOT/results/SUMMARY.md"
sed -n '1,200p' "$DS4_BENCH_ROOT/results/analysis/decisions.csv"
sed -n '1,160p' "$DS4_BENCH_ROOT/results/analysis/component-comparisons.csv"
( cd "$DS4_BENCH_ROOT" && \
  shasum -c laguna-m5-max-finalists-results.tar.gz.sha256 )
```

Choose a new tag. Start with `r1` for the date of the run and increment it if
either the Git tag or release already exists. `upload-draft` independently
fails closed on either collision.

```zsh
release_date=$(date +%F)
export DS4_RELEASE_TAG="bench-laguna-s2.1-m5-max-${release_date}-finalists-r1"
cd "$DS4_RUNBOOK_REPO"
benchmark/run_m5_finalists.zsh upload-draft
```

Return the verified draft URL, archive SHA-256, runbook/timed/trace/model
identities, complete decision table, and every reject/not-eligible path. Keep
`DS4_BENCH_ROOT` untouched until the reviewer confirms the release. Publication
and its post-publication tag check are specified in
[Upload a verified draft release](#upload-a-verified-draft-release).

The remainder of this file explains the protocol and provides reference
details. It does not add another execution pass: an agent that followed the
contract above must not repeat `prepare`, `prove`, `measure`, or `all` from the
sections below.

## What this run does

The prior discovery release is
[`bench-laguna-s2.1-m5-max-2026-08-10`](https://github.com/michaelasper/ds4/releases/tag/bench-laguna-s2.1-m5-max-2026-08-10).
Its raw archive SHA-256 is
`0298bf9b3ca01a2bdba44b0dd356156eacd0de26589fc51105136f844e053146`.
Its summary incorrectly skipped the finalist stage, but independent review of
the raw evidence selected these three cells:

| Candidate | Discovery estimate | Exact selector |
| --- | ---: | --- |
| `05-canonical-ladder` | +2.955% | `DS4_METAL_LAGUNA_DECODE_LADDER=7,15,23,31,39,47` |
| `12-gqa3` | +1.981% | `DS4_METAL_LAGUNA_SWA_GQA3=1` |
| `14-front-rung-ladder` | +2.331% | `DS4_METAL_LAGUNA_DECODE_LADDER=1,7,15,23,31,39,47` |

This document is a prospective amendment written before confirmation timing;
it does not pretend that the discovery runbook's broken finalist section was
executed. It replaces separate A/B loads with one shared-baseline complete
block, pre-registers both ladder×GQA3 combinations instead of adaptively
inventing one, and uses one 16-hypothesis promotion family. It omits the old
P8/P16 performance secondary because every selected optimisation is a
decode-route change; P16 logits still remain a mandatory parity gate. These
changes reduce model reloads, counterbalance order, and prevent post-result
choice while retaining the original D8 steady-decode target.

Discovery timing is selection evidence only. This confirmation run does not
reuse any discovery timing observation.

The confirmation matrix pre-registers six arms:

| Arm | Selectors beyond R1 |
| --- | --- |
| `00-r1-baseline` | none |
| `05-canonical-ladder` | canonical six-rung ladder |
| `12-gqa3` | GQA3 |
| `14-front-rung-ladder` | front seven-rung ladder |
| `15-canonical-ladder-gqa3` | canonical ladder + GQA3 |
| `16-front-rung-ladder-gqa3` | front ladder + GQA3 |

The two ladder schedules are alternatives. No arm enables both ladder values.
The combinations are pre-registered before timing, so there is no adaptive
second benchmark and no post-result selector invention.

Each of ten blocks runs all proof-and-warm-up-eligible arms once. The schedule
is a precomputed balanced, mirrored design whose ten block orders are shuffled
with the fixed seed `20260810`. Every surviving candidate runs before and after
the block-local R1 baseline exactly five times; every pair of arms also runs in
each relative order five times. The alternating Williams-style base order
spreads every possible directed predecessor→successor transition, with counts
differing by at most one. Position counts differ by at most two in the smallest
dynamic matrices. This is 60 D8 processes when all proofs pass,
instead of 100 processes from five separate A/B experiments.

The run intentionally does not repeat:

- the eleven discovery cells that missed the screen gate;
- the broken pre-rebase and clean-rebase controls;
- PR8 GPU sampling, SSD streaming, Q8 lm-head screening, Q4, or DFlash;
- a timed trace build or any trace-enabled timed arm.

## Power-mode policy

There is **no power-mode requirement or power-mode gate**.

- High Power Mode, Low Power Mode, AC/battery state, Energy Mode, display
  arrangement, and `pmset` thermal output never accept or reject an arm.
- The runner records sanitised host, power, display, and thermal snapshots for
  context only.
- There is no operator override file and no instruction to reinterpret
  `system_profiler` or `pmset` output.
- Use High Power Mode if that is the deployment configuration you want to
  measure. The protocol neither verifies nor relies on it.

The only host identity gate is Apple M5 Max with 128 GiB RAM. The memory gate
rejects a timed arm if measured pageouts, swapouts, or used swap growth above
1 MiB occur during that arm. Swapins are recorded but are not a gate.

## Immutable code and input identity

| Role | Commit |
| --- | --- |
| Timed R1 baseline and all timed candidates | `c8f25ee7ef1b6f16eff470ccf337d1546e314906` |
| Untimed completion-scoped route companion | `4322b2ca7664040b811de4426cc791cb499b2866` |

Every timed process enables `DS4_METAL_GLM_QMV_R1=1`. Candidate processes add
only the selector strings in the matrix above. The timed revision never sets a
trace selector.

The required model is:

```text
laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
SHA-256 61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903
```

The required prompt is `speed-bench/promessi_sposi.txt` from the timed commit:

```text
SHA-256 f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f
```

The driver refuses a different model, prompt, runbook checkout, code revision,
or built `ds4-bench` binary. It hashes the model once, pins its device, inode,
size, modification time, and change time, and checks that identity before and
after every process so a concurrent model replacement is fatal.

## Prepare the checkout

Use a clean checkout of the `michaelasper/ds4` fork. The runbook commit itself
must be pushed before benchmarking because the draft release targets that
immutable commit. The remote-agent contract already performs this step; do not
clone or prepare a second checkout during the same run. For a manual run that
starts at this section, use the same sequence:

```zsh
export DS4_RUNBOOK_REPO="$(mktemp -d /private/tmp/ds4-finalist-runbook.XXXXXX)"
git clone --branch laguna-s2.1 --single-branch \
  https://github.com/michaelasper/ds4.git "$DS4_RUNBOOK_REPO"
cd "$DS4_RUNBOOK_REPO"
runbook_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote https://github.com/michaelasper/ds4.git \
  refs/heads/laguna-s2.1 | awk '{print $1}')
[[ -n "$remote_sha" && "$runbook_sha" == "$remote_sha" ]]
git switch --detach "$runbook_sha"
[[ -z "$(git status --porcelain=v1)" ]]
```

The final check must succeed. The driver refuses a dirty runbook checkout.

Choose a fresh results root under `/private/tmp` and the existing model
directory. Do not put the large model inside the public results archive.

```zsh
export DS4_BENCH_ROOT="$(mktemp -d /private/tmp/ds4-laguna-finalists.XXXXXX)"
export DS4_GGUF_DIR=/absolute/path/to/laguna-models
export DS4_JOBS=8
export DS4_COOLDOWN_SECONDS=30
```

Requirements:

- `DS4_BENCH_ROOT` and `DS4_GGUF_DIR` must be absolute paths.
- `DS4_BENCH_ROOT` must be empty. Never reuse or overwrite a prior root.
- If the model is absent, its volume must have at least 55 GiB free before
  `prepare` starts.
- `DS4_COOLDOWN_SECONDS` is a fixed delay between measurement blocks, not a
  power or thermal gate.
- Keep the same terminal session for the whole run so the exported paths do
  not change.

If the model is absent, `prepare` invokes the repository's model downloader.
Hashing the approximately 45 GiB model happens once before any timed arm.

## Run the benchmark

The normal invocation is exactly:

```zsh
cd "$DS4_RUNBOOK_REPO"
benchmark/run_m5_finalists.zsh all
```

The phases run in this order:

1. `prepare`
   - captures the exact runbook commit and protocol-file hashes;
   - creates detached timed and trace worktrees;
   - verifies model and prompt hashes;
   - records sanitised host state without enforcing power mode;
   - builds both revisions in isolated environments;
   - runs the Metal and trace-transaction tests.
2. `prove`
   - captures baseline parity;
   - runs candidate parity for all five candidates;
   - runs one completion-scoped route proof per candidate;
   - marks a failed proof `SEMANTIC_REJECT` instead of silently falling back;
   - retains that candidate as `NOT_ELIGIBLE` evidence while allowing sound
     candidates to continue.
3. `measure`
   - performs one unanalysed warm-up per proof-eligible arm;
   - omits a warm-up reject with an explicit `NOT_ELIGIBLE` reason;
   - runs ten balanced, mirrored complete blocks using seed `20260810`;
   - records command, revision, model, selector, CSV, stdout, stderr, hashes,
     maximum RSS, before/after state, and status for every process;
   - scans diagnostics with Python and fails closed if the scanner errors.
4. `analyse`
   - uses paired `log(candidate/baseline)` values;
   - calculates an exact two-sided sign-flip test;
   - bootstraps the median log ratio for the 95% lower bound;
   - applies step-down Holm correction across one 16-hypothesis promotion
     family;
   - reports four GQA3×ladder component comparisons and their predeclared
     combination-winner gates;
   - generates `SUMMARY.md` and `RELEASE.md` from raw evidence.
5. `package`
   - deterministically reruns the archived analyser and byte-compares every
     generated report;
   - checks all status values and scans for common identity leaks;
   - creates a relative-path per-file `MANIFEST.sha256`;
   - creates the archive and a portable basename-only SHA sidecar;
   - strips owner identity, xattrs, ACLs, and file flags from tar headers;
   - extracts the archive into a fresh directory and verifies its exact file
     set, manifest, regular-file types, safe paths, and normalised ownership.

The workload for every timed arm is D8:

```text
model:       pinned Q2/Q3 model above
prompt:      promessi_sposi.txt
ctx-start:   8192
ctx-max:     8192
ctx-alloc:   8321
generation:  128 greedy tokens
primary:     gen_steady_tps
```

P16 logits are captured during parity. These selectors are decode-only, so
the run does not spend another 60 model loads on a prefill performance matrix.

### If the run is interrupted

Do not delete individual arms, rewrite `status.txt`, or rerun into the same
directory. Preserve the interrupted root for diagnosis and start `all` again
with a new `DS4_BENCH_ROOT`. This keeps block randomisation, filesystem state,
and evidence provenance unambiguous.

The phase commands exist for inspection and development:

```zsh
benchmark/run_m5_finalists.zsh prepare
benchmark/run_m5_finalists.zsh prove
benchmark/run_m5_finalists.zsh measure
benchmark/run_m5_finalists.zsh analyse
benchmark/run_m5_finalists.zsh package
```

For an official result, prefer `all` in one shell. Do not manually skip a
phase. A proof- or warm-up-rejected candidate is automatically omitted from
timing and retained as `NOT_ELIGIBLE` evidence. If no candidate remains, the
runner skips confirmation timing and still produces a complete no-promotion
summary and archive.

## Parity and route requirements

Every candidate must pass before it is timed:

- the complete 128-token generated byte stream matches R1 exactly;
- P16 frontier logits match R1 bit for bit;
- ladder-only one-row decode logits match R1 bit for bit;
- GQA3-containing one-row decode logits have the same finite argmax,
  maximum absolute difference at most `5e-4`, and RMS at most `1e-4`;
- the route companion reports `completion=waited`;
- GQA3-containing arms report exactly 36 GQA3 SWA layers and 12 global grouped
  layers;
- the canonical ladder reports layers `7,15,23,31,39,47`, six flushes, and
  waited completion;
- the front ladder reports layers `1,7,15,23,31,39,47`, seven flushes, and
  waited completion.

Generated output extraction preserves the whole byte sequence between the
single `decoded text` marker and its final delimiter. It does not compare only
the first line.

## Statistical decision rules

The three selected single cells remain members of the original 14-cell family;
the eleven unconfirmed discovery cells receive `p=1`. The two pre-registered
combinations join those 14 in one 16-hypothesis promotion family. One
step-down Holm correction, with the required cumulative maximum, therefore
controls selection of the final promotion winner across singles and
combinations. The four combination-versus-component comparisons form a
separate selection-safeguard family: they cannot make a candidate pass against
R1, but a combination must pass both of its component gates before it can be
named the winner.

For each candidate, the exact two-sided sign-flip test enumerates all `2^10`
sign assignments and uses the absolute mean paired log ratio as its statistic.
The confidence bound resamples the ten paired log ratios 20,000 times with a
fixed candidate-specific seed, takes the 2.5th percentile of the bootstrap
median log ratio, and converts that bound back to percent. The reported median
effect is likewise the median log ratio converted back to percent.

A candidate is accepted only if all of these hold:

- exactly ten valid paired blocks;
- median steady-decode improvement at least `+1.5%`;
- bootstrap 95% lower bound for that median above `+1.0%`;
- Holm-adjusted exact sign-flip `p <= .05`;
- at least eight of ten paired effects are positive;
- candidate parity is `PASS`;
- route proof is `PASS`;
- every timed candidate and paired baseline arm is `PASS`.

The highest-median accepted candidate is the promotion result. Combination
component rows separately compare each combination with both components. A
combination can outrank a simpler accepted component only when both rows have
ten valid pairs, positive median and 95% lower bound, four-member Holm-adjusted
`p <= .05`, and at least eight positive blocks. First-token latency remains in
every raw D8 CSV but is not copied into the promotion table. Recorded
power/thermal state is contextual only. Neither is a promotion gate.
An exact numerical tie is resolved by the pre-registered matrix order, which
lists single-selector arms before combinations.

No observation is dropped as an outlier. A missing CSV or malformed value is
not converted to zero; it makes that pair invalid and therefore fails the
ten-pair requirement.

## Inspect the local result

After `all` succeeds, the driver prints four absolute paths. Inspect them
before uploading:

```zsh
sed -n '1,240p' "$DS4_BENCH_ROOT/results/SUMMARY.md"
sed -n '1,200p' "$DS4_BENCH_ROOT/results/analysis/decisions.csv"
sed -n '1,160p' "$DS4_BENCH_ROOT/results/analysis/component-comparisons.csv"

( cd "$DS4_BENCH_ROOT" && \
  shasum -c laguna-m5-max-finalists-results.tar.gz.sha256 )
```

The SHA command is portable because its sidecar contains only
`laguna-m5-max-finalists-results.tar.gz`, not the original absolute path.

Required local deliverables:

```text
results/SUMMARY.md
results/RELEASE.md
results/MANIFEST.sha256
results/metadata/BENCHMARK.md
results/metadata/runbook-revision.txt
laguna-m5-max-finalists-results.tar.gz
laguna-m5-max-finalists-results.tar.gz.sha256
```

Do not report a tokens/second headline without `SUMMARY.md` and the raw
archive.

## Upload a verified draft release

Choose a new, unused tag. Never replace or add assets to the discovery tag.
Include `r2`, `r3`, and so on if the same calendar date already has a result.

```zsh
release_date=$(date +%F)
export DS4_RELEASE_TAG="bench-laguna-s2.1-m5-max-${release_date}-finalists-r1"
cd "$DS4_RUNBOOK_REPO"
benchmark/run_m5_finalists.zsh upload-draft
```

`upload-draft` performs these external actions:

1. refuses an existing tag;
2. creates a draft prerelease targeted at the exact runbook commit;
3. uploads the raw archive, portable archive SHA, summary, per-file manifest,
   exact `BENCHMARK.md`, and runbook-commit file;
4. downloads the draft assets into a fresh directory;
5. byte-compares all six downloaded assets with their local sources and
   verifies the archive SHA and tar headers;
6. asserts the exact six-asset inventory, unused remote tag, exact draft
   `targetCommitish`, runbook commit, draft state, and prerelease state;
7. prints the verified draft URL and the exact publish command.

If `upload-draft` fails after GitHub has created the draft, do not blindly
rerun it. Inspect or delete the partial draft, choose a fresh unused tag, and
run `upload-draft` again; the driver intentionally refuses to mutate an
existing release.

Read the downloaded/verified `SUMMARY.md`. If it agrees with the raw decision
ledger, publish the already-verified draft:

```zsh
gh release edit "$DS4_RELEASE_TAG" \
  --repo michaelasper/ds4 \
  --draft=false \
  --prerelease
```

Then record the final URL:

```zsh
gh release view "$DS4_RELEASE_TAG" \
  --repo michaelasper/ds4 \
  --json url,tagName,targetCommitish,isDraft,isPrerelease,assets

git ls-remote --tags https://github.com/michaelasper/ds4.git \
  "refs/tags/$DS4_RELEASE_TAG" "refs/tags/$DS4_RELEASE_TAG^{}"
```

GitHub may defer creation of a draft release's Git tag until publication.
After publication, the resolved tag (the peeled `^{}` value for an annotated
tag, otherwise the direct value) must equal the archived runbook commit.

The release target identifies the runbook, while `SUMMARY.md` separately
identifies the timed commit, trace commit, model, prompt, and discovery input.
This prevents a release tag from implying that the timing binary and protocol
are the same commit.

## Return contract

Return all of the following to the reviewer:

- release URL and tag;
- archive SHA-256;
- exact runbook commit;
- timed and trace commits;
- model SHA-256;
- the complete `SUMMARY.md` decision table;
- whether the draft download verification passed;
- any `SEMANTIC_REJECT` or `NOT_ELIGIBLE` paths.

Keep `DS4_BENCH_ROOT` until the reviewer confirms that the public archive and
manifest verify. The public release is a transport copy; the untouched local
root remains the recovery source.
