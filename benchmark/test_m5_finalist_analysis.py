#!/usr/bin/env python3

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import random
import shutil


BASELINE = "00-r1-baseline"
CANDIDATES = {
    "05-canonical-ladder": 103.0,
    "12-gqa3": 102.0,
    "14-front-rung-ladder": 102.5,
    "15-canonical-ladder-gqa3": 105.0,
    "16-front-rung-ladder-gqa3": 104.5,
}


class FinalistAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "results"
        self.root.mkdir()
        metadata = self.root / "metadata"
        metadata.mkdir()
        values = {
            "runbook-revision.txt": "a" * 40,
            "timed-revision.txt": "b" * 40,
            "trace-revision.txt": "c" * 40,
            "model-sha256.txt": "d" * 64,
            "prompt-sha256.txt": "e" * 64,
            "discovery-release.txt": "discovery-tag",
            "discovery-archive-sha256.txt": "f" * 64,
            "timed-binary-sha256.txt": "1" * 64,
            "trace-binary-sha256.txt": "2" * 64,
            "cooldown-seconds.txt": "30",
        }
        for name, value in values.items():
            (metadata / name).write_text(value + "\n")
        for candidate in CANDIDATES:
            for category in ("parity", "route-proof"):
                directory = self.root / category / candidate
                directory.mkdir(parents=True)
                (directory / "status.txt").write_text("PASS\n")
        for block in range(1, 11):
            base = self.root / "confirmation" / f"block-{block:02d}" / BASELINE
            self.write_arm(base, 100.0)
            for candidate, value in CANDIDATES.items():
                self.write_arm(base.parent / candidate, value)
        (metadata / "random-seed.txt").write_text("20260810\n")
        self.write_order([BASELINE, *CANDIDATES])

    def write_order(self, canonical_arms):
        (self.root / "confirmation" / "active-arms.txt").write_text(
            "\n".join(canonical_arms) + "\n"
        )
        order_path = self.root / "confirmation" / "balanced-order.csv"
        pattern = [0]
        low, high = 1, len(canonical_arms) - 1
        while len(pattern) < len(canonical_arms):
            pattern.append(low)
            low += 1
            if len(pattern) < len(canonical_arms):
                pattern.append(high)
                high -= 1
        orders = []
        for shift in range(5):
            order = [
                canonical_arms[(index + shift) % len(canonical_arms)]
                for index in pattern
            ]
            orders.extend((order, list(reversed(order))))
        random.Random(20260810).shuffle(orders)
        with order_path.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["block", "position", "arm"])
            for block, ordered in enumerate(orders, 1):
                for position, arm in enumerate(ordered, 1):
                    writer.writerow([block, position, arm])

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_arm(directory: Path, value: float):
        directory.mkdir(parents=True)
        (directory / "status.txt").write_text("PASS\n")
        with (directory / "d8.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["gen_steady_tps"])
            writer.writerow([value])

    def run_analysis(self):
        analyser = Path(__file__).with_name("m5_finalist_analysis.py")
        subprocess.run([sys.executable, str(analyser), str(self.root)], check=True)
        with (self.root / "analysis" / "decisions.csv").open(newline="") as handle:
            return {row["cell"]: row for row in csv.DictReader(handle)}

    def test_constant_positive_blocks_pass_all_gates(self):
        decisions = self.run_analysis()
        self.assertEqual(set(decisions), set(CANDIDATES))
        for row in decisions.values():
            self.assertEqual(row["n"], "10")
            self.assertEqual(row["positive_blocks"], "10")
            self.assertEqual(row["decision"], "ACCEPT")
            self.assertLessEqual(float(row["holm_p_adj"]), 0.05)
        self.assertAlmostEqual(
            float(decisions["05-canonical-ladder"]["holm_p_adj"]),
            16 * (2 / 1024),
        )
        self.assertAlmostEqual(
            float(decisions["15-canonical-ladder-gqa3"]["holm_p_adj"]),
            16 * (2 / 1024),
        )
        summary = (self.root / "SUMMARY.md").read_text()
        self.assertIn("15-canonical-ladder-gqa3", summary)
        self.assertIn("Power mode is not a gate", summary)
        self.assertIn("discovery-tag", summary)
        with (self.root / "analysis" / "component-comparisons.csv").open(newline="") as handle:
            comparisons = list(csv.DictReader(handle))
        self.assertEqual(len(comparisons), 4)
        self.assertTrue(all(row["decision"] == "PASS" for row in comparisons))

    def test_missing_block_is_a_reject_not_a_zero(self):
        missing = (
            self.root
            / "confirmation"
            / "block-10"
            / "14-front-rung-ladder"
            / "d8.csv"
        )
        missing.unlink()
        decisions = self.run_analysis()
        row = decisions["14-front-rung-ladder"]
        self.assertEqual(row["n"], "9")
        self.assertEqual(row["decision"], "REJECT")
        self.assertIn("expected 10", row["reason"])

    def test_not_eligible_candidate_is_omitted_but_reported(self):
        omitted = "16-front-rung-ladder-gqa3"
        directory = self.root / "not-eligible" / omitted
        directory.mkdir(parents=True)
        (directory / "status.txt").write_text("NOT_ELIGIBLE\n")
        (directory / "reason.txt").write_text("route proof rejected\n")
        for block in range(1, 11):
            shutil.rmtree(
                self.root / "confirmation" / f"block-{block:02d}" / omitted
            )
        active = [BASELINE, *(cell for cell in CANDIDATES if cell != omitted)]
        self.write_order(active)
        decisions = self.run_analysis()
        self.assertEqual(decisions[omitted]["n"], "0")
        self.assertEqual(decisions[omitted]["decision"], "REJECT")
        self.assertIn("NOT_ELIGIBLE: route proof rejected", decisions[omitted]["reason"])
        for cell in CANDIDATES:
            if cell != omitted:
                self.assertEqual(decisions[cell]["decision"], "ACCEPT")
        summary = (self.root / "SUMMARY.md").read_text()
        self.assertIn("route proof rejected", summary)

    def test_no_eligible_candidate_produces_no_promotion_summary(self):
        for candidate in CANDIDATES:
            directory = self.root / "not-eligible" / candidate
            directory.mkdir(parents=True)
            (directory / "status.txt").write_text("NOT_ELIGIBLE\n")
            (directory / "reason.txt").write_text("proof rejected\n")
        for block in range(1, 11):
            shutil.rmtree(self.root / "confirmation" / f"block-{block:02d}")
        (self.root / "confirmation" / "active-arms.txt").write_text(BASELINE + "\n")
        with (self.root / "confirmation" / "balanced-order.csv").open(
            "w", newline=""
        ) as handle:
            csv.writer(handle, lineterminator="\n").writerow(
                ["block", "position", "arm"]
            )
        decisions = self.run_analysis()
        self.assertTrue(all(row["decision"] == "REJECT" for row in decisions.values()))
        summary = (self.root / "SUMMARY.md").read_text()
        self.assertIn("No candidate passed every predeclared gate", summary)

    def test_combo_cannot_win_when_it_loses_to_a_component(self):
        replacements = {
            "15-canonical-ladder-gqa3": 102.5,
            "16-front-rung-ladder-gqa3": 102.4,
        }
        for block in range(1, 11):
            for candidate, value in replacements.items():
                directory = (
                    self.root
                    / "confirmation"
                    / f"block-{block:02d}"
                    / candidate
                )
                with (directory / "d8.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(["gen_steady_tps"])
                    writer.writerow([value])
        decisions = self.run_analysis()
        self.assertEqual(decisions["15-canonical-ladder-gqa3"]["decision"], "ACCEPT")
        summary = (self.root / "SUMMARY.md").read_text()
        self.assertIn("**05-canonical-ladder**", summary)
        with (
            self.root / "analysis" / "component-comparisons.csv"
        ).open(newline="") as handle:
            comparisons = list(csv.DictReader(handle))
        self.assertTrue(
            any(
                row["combo"] == "15-canonical-ladder-gqa3"
                and row["decision"] == "FAIL"
                for row in comparisons
            )
        )


if __name__ == "__main__":
    unittest.main()
