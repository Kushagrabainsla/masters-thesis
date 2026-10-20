"""Unit tests for the E-31 state model."""

from __future__ import annotations

import unittest

from model import Policy, SessionRuntime, token_digest
from run_e31 import bounded_model_check, run_source_scenarios


class SessionRuntimeTest(unittest.TestCase):
    def runtime(self, policy: Policy) -> SessionRuntime:
        return SessionRuntime(policy=policy, initial_tokens=tuple([10] * 15 + [28]))

    def kinds(self, runtime: SessionRuntime) -> set[str]:
        return {violation["kind"] for violation in runtime.violations}

    def test_replace_retains_stale_hash_only_for_mutable_policy(self) -> None:
        mutable = self.runtime(Policy.MUTABLE)
        original_key = mutable.block_key
        mutable.replace_last(11)
        mutable.schedule()
        mutable.complete_oldest()
        mutable.publish_cache()
        self.assertEqual(original_key, mutable.block_key)
        self.assertIn("cache_key_payload_mismatch", self.kinds(mutable))

        hash_only = self.runtime(Policy.HASH_ONLY)
        hash_only.replace_last(11)
        hash_only.schedule()
        hash_only.complete_oldest()
        hash_only.publish_cache()
        self.assertEqual(token_digest(hash_only.tokens), hash_only.block_key)
        self.assertNotIn("cache_key_payload_mismatch", self.kinds(hash_only))

    def test_hash_invalidation_does_not_fence_old_work(self) -> None:
        runtime = self.runtime(Policy.HASH_ONLY)
        runtime.schedule()
        runtime.append_input(13)
        runtime.complete_oldest()
        self.assertIn("stale_work_applied", self.kinds(runtime))

    def test_version_fence_discards_old_work(self) -> None:
        runtime = self.runtime(Policy.VERSIONED)
        runtime.schedule()
        runtime.append_input(13)
        runtime.complete_oldest()
        self.assertFalse(runtime.violations)
        self.assertEqual(1, runtime.discarded_work)

    def test_version_fence_discards_old_parser_event(self) -> None:
        runtime = self.runtime(Policy.VERSIONED)
        runtime.schedule()
        runtime.complete_oldest()
        runtime.append_input(13)
        runtime.flush_oldest()
        self.assertFalse(runtime.violations)
        self.assertEqual(1, runtime.discarded_events)

    def test_version_fence_orders_parser_events(self) -> None:
        mutable = self.runtime(Policy.MUTABLE)
        mutable.schedule()
        mutable.schedule()
        mutable.complete_oldest()
        mutable.complete_oldest()
        mutable.flush_newest()
        self.assertIn("out_of_order_event_emitted", self.kinds(mutable))

        versioned = self.runtime(Policy.VERSIONED)
        versioned.schedule()
        versioned.schedule()
        versioned.complete_oldest()
        versioned.complete_oldest()
        versioned.flush_newest()
        self.assertFalse(versioned.violations)
        self.assertEqual(1, versioned.discarded_events)

    def test_full_abort_clears_old_callbacks(self) -> None:
        runtime = self.runtime(Policy.ABORT_REBUILD)
        runtime.schedule()
        runtime.append_input(13)
        runtime.complete_oldest()
        runtime.flush_oldest()
        self.assertFalse(runtime.violations)
        self.assertEqual([], runtime.queue)
        self.assertEqual([], runtime.pending)
        self.assertGreater(runtime.rebuild_tokens, 0)

    def test_synchronous_policy_drains_before_update(self) -> None:
        runtime = self.runtime(Policy.SYNCHRONOUS)
        runtime.schedule()
        runtime.append_input(13)
        self.assertFalse(runtime.violations)
        self.assertEqual(1, runtime.drained_work)
        self.assertEqual(1, len(runtime.emitted))

    def test_separate_frontiers_preserve_logical_history(self) -> None:
        hash_only = self.runtime(Policy.HASH_ONLY)
        before = hash_only.tokens
        hash_only.preempt()
        hash_only.fold_after_preempt()
        self.assertNotEqual(before, hash_only.tokens)
        self.assertIn("logical_history_lost", self.kinds(hash_only))

        versioned = self.runtime(Policy.VERSIONED)
        before = versioned.tokens
        versioned.preempt()
        versioned.fold_after_preempt()
        self.assertEqual(before, versioned.tokens)
        self.assertFalse(versioned.violations)

    def test_epoch_only_does_not_separate_progress_frontiers(self) -> None:
        runtime = self.runtime(Policy.EPOCH_ONLY)
        runtime.preempt()
        runtime.fold_after_preempt()
        self.assertIn("logical_history_lost", self.kinds(runtime))

    def test_source_matrix_has_all_policy_scenario_pairs(self) -> None:
        rows, _ = run_source_scenarios()
        self.assertEqual(24, len(rows))
        pairs = {(row["scenario"], row["policy"]) for row in rows}
        self.assertEqual(24, len(pairs))

    def test_bounded_check_is_deterministic(self) -> None:
        first, _ = bounded_model_check(3)
        second, _ = bounded_model_check(3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
