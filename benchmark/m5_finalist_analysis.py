#!/usr/bin/env python3
"""Analyse the pre-registered Laguna M5 Max finalist confirmation matrix."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path
import random
import statistics
import sys


BASELINE = "00-r1-baseline"
CANDIDATES = (
    "05-canonical-ladder",
    "12-gqa3",
    "14-front-rung-ladder",
    "15-canonical-ladder-gqa3",
    "16-front-rung-ladder-gqa3",
)
SINGLE_FAMILY = (
    "01-gqa9",
    "02-fused-router",
    "03-residual-fusion",
    "04-dense-q8-decode",
    "05-canonical-ladder",
    "06-direct-kv-prefill",
    "07-batched-dense-q8-prefill",
    "08-router-simd-topk",
    "09-simd32-qk",
    "10-rope-atlas",
    "11-simd32-atlas",
    "12-gqa3",
    "13-moe-threshold32",
    "14-front-rung-ladder",
)
COMBO_FAMILY = (
    "15-canonical-ladder-gqa3",
    "16-front-rung-ladder-gqa3",
)
PROMOTION_FAMILY = SINGLE_FAMILY + COMBO_FAMILY
COMPONENTS = {
    "15-canonical-ladder-gqa3": ("05-canonical-ladder", "12-gqa3"),
    "16-front-rung-ladder-gqa3": ("14-front-rung-ladder", "12-gqa3"),
}


def read_status(directory: Path) -> str:
    try:
        return (directory / "status.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return "MISSING"


def metric(directory: Path, key: str = "gen_steady_tps") -> float:
    csv_path = directory / "d8.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or key not in rows[-1]:
        raise ValueError(f"missing {key} in {csv_path}")
    value = float(rows[-1][key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {key}={value!r} in {csv_path}")
    return value


def exact_signflip(log_ratios: list[float]) -> float:
    if len(log_ratios) != 10:
        return 1.0
    observed = abs(statistics.mean(log_ratios))
    hits = 0
    for mask in range(1 << len(log_ratios)):
        permuted = statistics.mean(
            value if (mask >> index) & 1 else -value
            for index, value in enumerate(log_ratios)
        )
        hits += abs(permuted) >= observed - 1e-15
    return hits / (1 << len(log_ratios))


def bootstrap_median_lower(log_ratios: list[float], seed: int) -> float:
    if len(log_ratios) != 10:
        return float("-inf")
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choice(log_ratios) for _ in log_ratios)
        for _ in range(20_000)
    ]
    medians.sort()
    return medians[499]


def percent(log_ratio: float) -> float:
    return 100.0 * math.expm1(log_ratio)


def holm_adjust(rows: list[dict[str, object]], family_size: int) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["p_value"]),
            str(row.get("cell", f"{row.get('combo')}:{row.get('component')}")),
        ),
    )
    running = 0.0
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, float(row["p_value"]) * (family_size - rank))
        running = max(running, adjusted)
        row["holm_p_adj"] = running


def probe_ok(root: Path, category: str, cell: str) -> bool:
    return read_status(root / category / cell) == "PASS"


def analyse_candidate(root: Path, cell: str, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    not_eligible = root / "not-eligible" / cell
    if read_status(not_eligible) == "NOT_ELIGIBLE":
        try:
            omitted_reason = (not_eligible / "reason.txt").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            omitted_reason = "reason ledger missing"
        return (
            {
                "cell": cell,
                "n": 0,
                "median_effect_pct": -100.0,
                "lower_ci_pct": -100.0,
                "p_value": 1.0,
                "holm_p_adj": 1.0,
                "positive_blocks": 0,
                "parity": "PASS" if probe_ok(root, "parity", cell) else "FAIL",
                "route_proof": "PASS" if probe_ok(root, "route-proof", cell) else "FAIL",
                "decision": "PENDING",
                "reason": "",
                "_failures": [f"NOT_ELIGIBLE: {omitted_reason}"],
            },
            [],
        )
    observations: list[dict[str, object]] = []
    log_ratios: list[float] = []
    failures: list[str] = []
    for block in range(1, 11):
        base_dir = root / "confirmation" / f"block-{block:02d}" / BASELINE
        candidate_dir = root / "confirmation" / f"block-{block:02d}" / cell
        base_status = read_status(base_dir)
        candidate_status = read_status(candidate_dir)
        try:
            base_value = metric(base_dir)
            candidate_value = metric(candidate_dir)
            log_ratio = math.log(candidate_value / base_value)
        except (OSError, ValueError, KeyError, IndexError) as error:
            base_value = float("nan")
            candidate_value = float("nan")
            log_ratio = float("nan")
            failures.append(f"block {block}: {error}")
        if base_status != "PASS" or candidate_status != "PASS":
            failures.append(
                f"block {block}: baseline={base_status}, candidate={candidate_status}"
            )
            for label, directory, value in (
                ("baseline", base_dir, base_status),
                ("candidate", candidate_dir, candidate_status),
            ):
                reason_path = directory / "reject-reason.txt"
                if value == "SEMANTIC_REJECT" and reason_path.is_file():
                    failures.append(
                        f"block {block} {label}: "
                        f"{reason_path.read_text(encoding='utf-8', errors='replace').strip()}"
                    )
        if math.isfinite(log_ratio) and base_status == "PASS" and candidate_status == "PASS":
            log_ratios.append(log_ratio)
        observations.append(
            {
                "cell": cell,
                "block": block,
                "baseline_status": base_status,
                "candidate_status": candidate_status,
                "baseline_gen_steady_tps": base_value,
                "candidate_gen_steady_tps": candidate_value,
                "log_ratio": log_ratio,
                "effect_pct": percent(log_ratio) if math.isfinite(log_ratio) else float("nan"),
            }
        )

    median_log = statistics.median(log_ratios) if log_ratios else float("-inf")
    lower_log = bootstrap_median_lower(log_ratios, seed)
    parity = probe_ok(root, "parity", cell)
    route = probe_ok(root, "route-proof", cell)
    row: dict[str, object] = {
        "cell": cell,
        "n": len(log_ratios),
        "median_effect_pct": percent(median_log),
        "lower_ci_pct": percent(lower_log),
        "p_value": exact_signflip(log_ratios),
        "holm_p_adj": 1.0,
        "positive_blocks": sum(value > 0 for value in log_ratios),
        "parity": "PASS" if parity else "FAIL",
        "route_proof": "PASS" if route else "FAIL",
        "decision": "PENDING",
        "reason": "",
    }
    row["_failures"] = failures
    return row, observations


def component_rows(raw: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for combo, components in COMPONENTS.items():
        combo_by_block = {int(row["block"]): row for row in raw.get(combo, [])}
        for component in components:
            component_by_block = {
                int(row["block"]): row for row in raw.get(component, [])
            }
            values: list[float] = []
            for block in range(1, 11):
                combo_row = combo_by_block.get(block)
                component_row = component_by_block.get(block)
                if not combo_row or not component_row:
                    continue
                if (
                    combo_row["baseline_status"] != "PASS"
                    or combo_row["candidate_status"] != "PASS"
                    or component_row["baseline_status"] != "PASS"
                    or component_row["candidate_status"] != "PASS"
                ):
                    continue
                combo_value = float(combo_row["candidate_gen_steady_tps"])
                component_value = float(component_row["candidate_gen_steady_tps"])
                if combo_value > 0 and component_value > 0:
                    values.append(math.log(combo_value / component_value))
            median_log = statistics.median(values) if values else float("-inf")
            lower_log = bootstrap_median_lower(values, 90_000 + len(result))
            result.append(
                {
                    "combo": combo,
                    "component": component,
                    "n": len(values),
                    "median_increment_pct": percent(median_log),
                    "lower_ci_pct": percent(lower_log),
                    "p_value": exact_signflip(values),
                    "holm_p_adj": 1.0,
                    "positive_blocks": sum(value > 0 for value in values),
                    "decision": "PENDING",
                    "reason": "",
                }
            )
    holm_adjust(result, len(result))
    for row in result:
        reasons = []
        if int(row["n"]) != 10:
            reasons.append(f"valid pairs={row['n']}, expected 10")
        if float(row["median_increment_pct"]) <= 0:
            reasons.append("median increment is not positive")
        if float(row["lower_ci_pct"]) <= 0:
            reasons.append("95% lower bound is not positive")
        if float(row["holm_p_adj"]) > 0.05:
            reasons.append("Holm-adjusted p exceeds .05")
        if int(row["positive_blocks"]) < 8:
            reasons.append("fewer than 8/10 blocks favour the combination")
        row["decision"] = "PASS" if not reasons else "FAIL"
        row["reason"] = "; ".join(reasons)
    return result


def promotion_champion(
    decisions: list[dict[str, object]], comparisons: list[dict[str, object]]
) -> dict[str, object] | None:
    accepted = [row for row in decisions if row["decision"] == "ACCEPT"]
    component_pass = {
        combo: all(
            row["decision"] == "PASS"
            for row in comparisons
            if row["combo"] == combo
        )
        and sum(1 for row in comparisons if row["combo"] == combo) == 2
        for combo in COMBO_FAMILY
    }
    eligible = [
        row
        for row in accepted
        if row["cell"] not in COMBO_FAMILY or component_pass.get(str(row["cell"]), False)
    ]
    # max() is stable: an exact numerical tie uses the pre-registered matrix
    # order, which lists single-selector arms before combinations.
    return max(eligible, key=lambda row: float(row["median_effect_pct"]), default=None)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_value(root: Path, name: str, fallback: str = "missing") -> str:
    try:
        value = (root / "metadata" / name).read_text(encoding="utf-8").strip()
        return value or fallback
    except OSError:
        return fallback


def validate_balanced_order(root: Path) -> list[str]:
    order_path = root / "confirmation" / "balanced-order.csv"
    try:
        with order_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seed = int(metadata_value(root, "random-seed.txt"))
        active_arms = [
            line.strip()
            for line in (root / "confirmation" / "active-arms.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, TypeError) as error:
        return [f"balanced-order ledger is unreadable: {error}"]
    failures: list[str] = []
    if (
        not active_arms
        or active_arms[0] != BASELINE
        or len(active_arms) != len(set(active_arms))
        or not set(active_arms[1:]).issubset(CANDIDATES)
    ):
        return [f"invalid active-arm ledger: {active_arms}"]
    expected_active = [
        BASELINE,
        *(
            cell
            for cell in CANDIDATES
            if read_status(root / "not-eligible" / cell) != "NOT_ELIGIBLE"
        ),
    ]
    if active_arms != expected_active:
        return [
            f"active-arm ledger {active_arms} does not match eligibility {expected_active}"
        ]
    if len(active_arms) == 1:
        if rows:
            failures.append("no candidate was active but timed order rows exist")
        return failures
    by_block: dict[int, list[tuple[int, str]]] = {}
    try:
        for row in rows:
            by_block.setdefault(int(row["block"]), []).append(
                (int(row["position"]), row["arm"])
            )
    except (KeyError, TypeError, ValueError) as error:
        return [f"randomisation ledger is malformed: {error}"]
    pattern = [0]
    low, high = 1, len(active_arms) - 1
    while len(pattern) < len(active_arms):
        pattern.append(low)
        low += 1
        if len(pattern) < len(active_arms):
            pattern.append(high)
            high -= 1
    base_orders: list[list[str]] = []
    for shift in range(5):
        order = [
            active_arms[(index + shift) % len(active_arms)] for index in pattern
        ]
        base_orders.append(order)
        base_orders.append(list(reversed(order)))
    random.Random(seed).shuffle(base_orders)
    position_counts = {
        arm: {position: 0 for position in range(1, len(active_arms) + 1)}
        for arm in active_arms
    }
    pair_order_counts = {
        (left, right): [0, 0]
        for left_index, left in enumerate(active_arms)
        for right in active_arms[left_index + 1 :]
    }
    transition_counts = {
        (left, right): 0 for left in active_arms for right in active_arms if left != right
    }
    for block in range(1, 11):
        positioned = sorted(by_block.get(block, []))
        positions = [position for position, _ in positioned]
        arms = [arm for _, arm in positioned]
        if positions != list(range(1, len(arms) + 1)):
            failures.append(f"block {block}: positions are not contiguous")
        if len(arms) != len(set(arms)):
            failures.append(f"block {block}: duplicate arm")
        if set(arms) != set(active_arms):
            failures.append(f"block {block}: arm set does not match active-arm ledger")
        expected = base_orders[block - 1]
        if arms != expected:
            failures.append(f"block {block}: order does not match seed {seed}")
        for position, arm in enumerate(arms, 1):
            if arm in position_counts and position in position_counts[arm]:
                position_counts[arm][position] += 1
        for pair, counts in pair_order_counts.items():
            left, right = pair
            if left in arms and right in arms:
                counts[0 if arms.index(left) < arms.index(right) else 1] += 1
        for transition in zip(arms, arms[1:]):
            if transition in transition_counts:
                transition_counts[transition] += 1
    if set(by_block) != set(range(1, 11)):
        failures.append("randomisation ledger does not contain exactly blocks 1..10")
    for arm, counts in position_counts.items():
        if max(counts.values()) - min(counts.values()) > 2:
            failures.append(f"{arm}: position counts are not balanced: {counts}")
    for pair, counts in pair_order_counts.items():
        if counts != [5, 5]:
            failures.append(f"{pair}: before/after counts are {counts}, expected [5, 5]")
    if transition_counts:
        values = list(transition_counts.values())
        if min(values) < 1 or max(values) - min(values) > 1:
            failures.append(f"directed transition counts are not spread 1-apart: {transition_counts}")
    return failures


def make_summary(root: Path, decisions: list[dict[str, object]], comparisons: list[dict[str, object]]) -> None:
    indexed = {str(row["cell"]): row for row in decisions}
    champion = promotion_champion(decisions, comparisons)
    summary = root / "SUMMARY.md"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("# Laguna S 2.1 M5 Max finalist confirmation\n\n")
        handle.write("Eligible candidates receive ten fresh balanced, mirrored blocks. No discovery-screen timing is reused.\n\n")
        handle.write("## Identity\n\n")
        handle.write(f"- Runbook commit: `{metadata_value(root, 'runbook-revision.txt')}`\n")
        handle.write(f"- Timed revision: `{metadata_value(root, 'timed-revision.txt')}`\n")
        handle.write(f"- Trace revision: `{metadata_value(root, 'trace-revision.txt')}`\n")
        handle.write(f"- Model SHA-256: `{metadata_value(root, 'model-sha256.txt')}`\n")
        handle.write(f"- Prompt SHA-256: `{metadata_value(root, 'prompt-sha256.txt')}`\n")
        handle.write(f"- Discovery release: `{metadata_value(root, 'discovery-release.txt')}`\n")
        handle.write(f"- Discovery archive SHA-256: `{metadata_value(root, 'discovery-archive-sha256.txt')}`\n")
        handle.write(f"- Balanced-order seed: `{metadata_value(root, 'random-seed.txt')}`\n")
        handle.write(f"- Fixed cooldown between blocks: `{metadata_value(root, 'cooldown-seconds.txt')}` seconds\n")
        handle.write(f"- Timed ds4-bench SHA-256: `{metadata_value(root, 'timed-binary-sha256.txt')}`\n")
        handle.write(f"- Trace ds4-bench SHA-256: `{metadata_value(root, 'trace-binary-sha256.txt')}`\n")
        handle.write("- Power mode is not a gate. Captured power data is informational only.\n\n")
        handle.write("## Decisions\n\n")
        handle.write("| Candidate | n | Median % | 95% lower % | raw p | Holm p | Positive blocks | Parity | Route | Decision |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for cell in CANDIDATES:
            row = indexed[cell]
            handle.write(
                f"| {cell} | {row['n']} | {float(row['median_effect_pct']):.4f} | "
                f"{float(row['lower_ci_pct']):.4f} | {float(row['p_value']):.6f} | "
                f"{float(row['holm_p_adj']):.6f} | {row['positive_blocks']} | "
                f"{row['parity']} | {row['route_proof']} | {row['decision']} |\n"
            )
        rejected = [row for row in decisions if row["reason"]]
        if rejected:
            handle.write("\n## Rejection reasons\n\n")
            for row in rejected:
                handle.write(f"- `{row['cell']}`: {row['reason']}\n")
        handle.write("\n## Combination component gates\n\n")
        handle.write("A combination can be the promotion winner only when it passes against both components.\n\n")
        handle.write("| Combination | Component | n | Median incremental % | 95% lower % | raw p | Holm p | Positive | Gate |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in comparisons:
            handle.write(
                f"| {row['combo']} | {row['component']} | {row['n']} | "
                f"{float(row['median_increment_pct']):.4f} | "
                f"{float(row['lower_ci_pct']):.4f} | {float(row['p_value']):.6f} | "
                f"{float(row['holm_p_adj']):.6f} | {row['positive_blocks']} | {row['decision']} |\n"
            )
        failed_comparisons = [row for row in comparisons if row["reason"]]
        if failed_comparisons:
            handle.write("\n### Component-gate reasons\n\n")
            for row in failed_comparisons:
                handle.write(
                    f"- `{row['combo']}` versus `{row['component']}`: {row['reason']}\n"
                )
        handle.write("\n## Promotion result\n\n")
        if champion:
            handle.write(
                f"The highest-median candidate that passed every predeclared gate is "
                f"**{champion['cell']}** at {float(champion['median_effect_pct']):.4f}%.\n"
            )
        else:
            handle.write("No candidate passed every predeclared gate. Do not promote an optimisation from this run.\n")
        handle.write("\n## Gates\n\n")
        handle.write("- Exactly ten valid paired blocks.\n")
        handle.write("- Median steady-decode improvement at least +1.5%.\n")
        handle.write("- Bootstrap 95% lower bound for the median above +1.0%.\n")
        handle.write("- Holm-adjusted two-sided exact sign-flip p at most .05.\n")
        handle.write("- At least eight of ten blocks positive.\n")
        handle.write("- Predeclared candidate parity and completion-scoped route proof PASS.\n")
        handle.write("- A combination must also pass both gates in the four-member component-comparison family to be champion.\n")
        handle.write("- No fallback/failure diagnostic and no measured pageout/swapout growth.\n")
        handle.write("- Power mode is recorded but never accepted, rejected, or overridden by the protocol.\n")


def make_release_notes(
    root: Path, decisions: list[dict[str, object]], comparisons: list[dict[str, object]]
) -> None:
    champion = promotion_champion(decisions, comparisons)
    path = root / "RELEASE.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("## Laguna S 2.1 M5 Max finalist confirmation\n\n")
        if champion:
            handle.write(
                f"Promotion result: **{champion['cell']}** passed all gates with a "
                f"{float(champion['median_effect_pct']):.4f}% median steady-decode improvement.\n\n"
            )
        else:
            handle.write("Promotion result: **no candidate passed every gate**.\n\n")
        handle.write("See `SUMMARY.md` for decisions and `MANIFEST.sha256` for every archived file.\n\n")
        handle.write(f"- Runbook commit: `{metadata_value(root, 'runbook-revision.txt')}`\n")
        handle.write(f"- Timed revision: `{metadata_value(root, 'timed-revision.txt')}`\n")
        handle.write(f"- Trace revision: `{metadata_value(root, 'trace-revision.txt')}`\n")
        handle.write(f"- Model SHA-256: `{metadata_value(root, 'model-sha256.txt')}`\n")
        handle.write(f"- Prompt SHA-256: `{metadata_value(root, 'prompt-sha256.txt')}`\n")
        handle.write(f"- Discovery release: `{metadata_value(root, 'discovery-release.txt')}`\n")
        handle.write(f"- Discovery archive SHA-256: `{metadata_value(root, 'discovery-archive-sha256.txt')}`\n")
        handle.write(f"- Balanced-order seed: `{metadata_value(root, 'random-seed.txt')}`\n")
        handle.write(f"- Timed ds4-bench SHA-256: `{metadata_value(root, 'timed-binary-sha256.txt')}`\n")
        handle.write(f"- Trace ds4-bench SHA-256: `{metadata_value(root, 'trace-binary-sha256.txt')}`\n")
        handle.write("- Power mode was not a benchmark gate.\n")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} RESULTS_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    decisions: list[dict[str, object]] = []
    all_observations: list[dict[str, object]] = []
    raw: dict[str, list[dict[str, object]]] = {}
    randomisation_failures = validate_balanced_order(root)
    for index, cell in enumerate(CANDIDATES):
        row, observations = analyse_candidate(root, cell, 20_260_810 + index)
        decisions.append(row)
        all_observations.extend(observations)
        raw[cell] = observations

    # The eleven unconfirmed discovery cells receive p=1 and sort after the
    # five measured rows. A single 16-member family controls the promotion
    # decision across three selected singles and two preregistered combos.
    holm_adjust(decisions, len(PROMOTION_FAMILY))

    for row in decisions:
        reasons: list[str] = list(row.pop("_failures")) + randomisation_failures
        if int(row["n"]) != 10:
            reasons.append(f"valid pairs={row['n']}, expected 10")
        if float(row["median_effect_pct"]) < 1.5:
            reasons.append("median improvement below +1.5%")
        if float(row["lower_ci_pct"]) <= 1.0:
            reasons.append("95% lower bound is not above +1.0%")
        if float(row["holm_p_adj"]) > 0.05:
            reasons.append("Holm-adjusted p exceeds .05")
        if int(row["positive_blocks"]) < 8:
            reasons.append("fewer than 8/10 blocks are positive")
        if row["parity"] != "PASS":
            reasons.append("parity did not pass")
        if row["route_proof"] != "PASS":
            reasons.append("route proof did not pass")
        row["decision"] = "ACCEPT" if not reasons else "REJECT"
        row["reason"] = "; ".join(reasons)

    observation_fields = [
        "cell", "block", "baseline_status", "candidate_status",
        "baseline_gen_steady_tps", "candidate_gen_steady_tps", "log_ratio", "effect_pct",
    ]
    decision_fields = [
        "cell", "n", "median_effect_pct", "lower_ci_pct", "p_value", "holm_p_adj",
        "positive_blocks", "parity", "route_proof", "decision", "reason",
    ]
    write_csv(analysis / "block-effects.csv", all_observations, observation_fields)
    write_csv(analysis / "decisions.csv", decisions, decision_fields)
    comparisons = component_rows(raw)
    write_csv(
        analysis / "component-comparisons.csv",
        comparisons,
        [
            "combo", "component", "n", "median_increment_pct", "lower_ci_pct",
            "p_value", "holm_p_adj", "positive_blocks", "decision", "reason",
        ],
    )
    make_summary(root, decisions, comparisons)
    make_release_notes(root, decisions, comparisons)
    for report in (
        analysis / "block-effects.csv",
        analysis / "decisions.csv",
        analysis / "component-comparisons.csv",
        root / "SUMMARY.md",
        root / "RELEASE.md",
    ):
        (report.with_name(report.name + ".sha256")).write_text(
            f"{file_sha256(report)}  {report.name}\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
