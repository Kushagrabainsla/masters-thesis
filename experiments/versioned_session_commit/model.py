"""Abstract state machine for mutable inference sessions.

This model does not implement vLLM. It isolates the state transitions that the
public issue reproducers expose. The experiment uses it as a mechanism screen.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Policy(StrEnum):
    MUTABLE = "mutable"
    HASH_ONLY = "hash_only"
    EPOCH_ONLY = "epoch_only"
    ABORT_REBUILD = "abort_rebuild"
    SYNCHRONOUS = "synchronous"
    VERSIONED = "versioned"


ALL_POLICIES = tuple(Policy)


def token_digest(tokens: tuple[int, ...]) -> str:
    payload = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def output_token(tokens: tuple[int, ...], revision: int, frontier: int) -> int:
    payload = f"{token_digest(tokens)}:{revision}:{frontier}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 32_000


@dataclass(frozen=True)
class Work:
    work_id: int
    revision: int
    incarnation: int
    frontier: int
    tokens: tuple[int, ...]
    output: int


@dataclass(frozen=True)
class PendingEvent:
    event_id: int
    revision: int
    incarnation: int
    frontier: int
    output: int


@dataclass
class SessionRuntime:
    """One session under one state-management policy."""

    policy: Policy
    initial_tokens: tuple[int, ...]
    session_id: int = 0
    tokens: tuple[int, ...] = field(init=False)
    revision: int = field(default=0, init=False)
    incarnation: int = field(default=0, init=False)
    frontier: int = field(default=0, init=False)
    schedule_frontier: int = field(default=0, init=False)
    emit_frontier: int = field(default=0, init=False)
    block_key: str | None = field(init=False)
    materialized_tokens: tuple[int, ...] | None = field(init=False)
    materialized_revision: int | None = field(default=0, init=False)
    preempted: bool = field(default=False, init=False)
    queue: list[Work] = field(default_factory=list, init=False)
    pending: list[PendingEvent] = field(default_factory=list, init=False)
    emitted: list[PendingEvent] = field(default_factory=list, init=False)
    violations: list[dict[str, Any]] = field(default_factory=list, init=False)
    history: list[str] = field(default_factory=list, init=False)
    next_work_id: int = field(default=0, init=False)
    next_event_id: int = field(default=0, init=False)
    operation_count: int = field(default=0, init=False)
    protocol_checks: int = field(default=0, init=False)
    discarded_work: int = field(default=0, init=False)
    discarded_events: int = field(default=0, init=False)
    rebuild_tokens: int = field(default=0, init=False)
    computed_tokens: int = field(default=0, init=False)
    drained_work: int = field(default=0, init=False)
    missing_callbacks: int = field(default=0, init=False)
    update_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.tokens = tuple(self.initial_tokens)
        self.block_key = token_digest(self.tokens)
        self.materialized_tokens = self.tokens

    def clone(self) -> SessionRuntime:
        return copy.deepcopy(self)

    def _step(self, name: str) -> None:
        self.operation_count += 1
        self.history.append(name)

    def _record(self, kind: str, **detail: Any) -> None:
        self.violations.append(
            {
                "kind": kind,
                "operation": self.operation_count,
                "revision": self.revision,
                "history": list(self.history),
                **detail,
            }
        )

    def _drain_before_update(self) -> None:
        while self.queue:
            self.complete_oldest(internal=True)
            self.drained_work += 1
        while self.pending:
            self.flush_oldest(internal=True)

    def _prepare_update(self) -> None:
        if self.policy is Policy.SYNCHRONOUS:
            self._drain_before_update()

    def _finish_update(self) -> None:
        self.revision += 1
        self.frontier = 0
        self.schedule_frontier = 0
        self.emit_frontier = 0
        self.update_count += 1
        self.preempted = False

        if self.policy is Policy.MUTABLE:
            return

        if self.policy is Policy.HASH_ONLY:
            self.block_key = None
            self.materialized_tokens = None
            self.materialized_revision = None
            return

        if self.policy is Policy.ABORT_REBUILD:
            self.incarnation += 1
            self.queue.clear()
            self.pending.clear()
            self.block_key = token_digest(self.tokens)
            self.materialized_tokens = self.tokens
            self.materialized_revision = self.revision
            self.rebuild_tokens += len(self.tokens)
            return

        if self.policy is Policy.SYNCHRONOUS:
            self.block_key = None
            self.materialized_tokens = None
            self.materialized_revision = None
            return

        if self.policy in (Policy.EPOCH_ONLY, Policy.VERSIONED):
            self.block_key = None
            self.materialized_tokens = None
            self.materialized_revision = None
            return

        raise AssertionError(f"unknown policy: {self.policy}")

    def append_input(self, token: int | None = None) -> None:
        self._step("append_input")
        self._prepare_update()
        value = token if token is not None else 1_000 + self.revision
        self.tokens = self.tokens + (value,)
        self._finish_update()

    def replace_last(self, token: int | None = None) -> None:
        self._step("replace_last")
        if not self.tokens:
            return
        self._prepare_update()
        value = token if token is not None else 2_000 + self.revision
        self.tokens = self.tokens[:-1] + (value,)
        self._finish_update()

    def schedule(self) -> None:
        self._step("schedule")
        work = Work(
            work_id=self.next_work_id,
            revision=self.revision,
            incarnation=self.incarnation,
            frontier=self.schedule_frontier,
            tokens=self.tokens,
            output=output_token(self.tokens, self.revision, self.schedule_frontier),
        )
        self.next_work_id += 1
        self.schedule_frontier += 1
        self.queue.append(work)

    def _work_is_current(self, work: Work) -> bool:
        if self.policy in (Policy.EPOCH_ONLY, Policy.VERSIONED):
            self.protocol_checks += 1
            return (
                work.revision == self.revision
                and work.incarnation == self.incarnation
                and work.frontier == self.frontier
            )
        if self.policy is Policy.ABORT_REBUILD:
            return (
                work.incarnation == self.incarnation
                and work.frontier == self.frontier
            )
        if self.policy is Policy.SYNCHRONOUS:
            return (
                work.revision == self.revision
                and work.frontier == self.frontier
            )
        return work.frontier == self.frontier

    def _complete(self, index: int, name: str, internal: bool = False) -> None:
        if not internal:
            self._step(name)
        if not self.queue:
            self.missing_callbacks += 1
            return
        work = self.queue.pop(index)
        self.computed_tokens += len(work.tokens) + 1

        if not self._work_is_current(work):
            self.discarded_work += 1
            return

        if work.revision != self.revision or work.frontier != self.frontier:
            self._record(
                "stale_work_applied",
                work_revision=work.revision,
                work_frontier=work.frontier,
                current_frontier=self.frontier,
            )

        event = PendingEvent(
            event_id=self.next_event_id,
            revision=work.revision,
            incarnation=work.incarnation,
            frontier=work.frontier,
            output=work.output,
        )
        self.next_event_id += 1
        self.pending.append(event)
        self.materialized_tokens = work.tokens
        self.materialized_revision = work.revision
        if self.block_key is None:
            self.block_key = token_digest(work.tokens)
        self.frontier += 1

    def complete_oldest(self, internal: bool = False) -> None:
        self._complete(0, "complete_oldest", internal)

    def complete_newest(self) -> None:
        self._complete(-1, "complete_newest")

    def _event_is_current(self, event: PendingEvent) -> bool:
        if self.policy in (Policy.EPOCH_ONLY, Policy.VERSIONED):
            self.protocol_checks += 1
            return (
                event.revision == self.revision
                and event.incarnation == self.incarnation
                and event.frontier == self.emit_frontier
            )
        if self.policy is Policy.ABORT_REBUILD:
            return (
                event.incarnation == self.incarnation
                and event.frontier == self.emit_frontier
            )
        if self.policy is Policy.SYNCHRONOUS:
            return (
                event.revision == self.revision
                and event.frontier == self.emit_frontier
            )
        return True

    def _flush(self, index: int, name: str, internal: bool = False) -> None:
        if not internal:
            self._step(name)
        if not self.pending:
            self.missing_callbacks += 1
            return
        event = self.pending.pop(index)
        if not self._event_is_current(event):
            self.discarded_events += 1
            return
        if event.revision != self.revision:
            self._record(
                "stale_event_emitted",
                event_revision=event.revision,
            )
        if event.frontier != self.emit_frontier:
            self._record(
                "out_of_order_event_emitted",
                event_frontier=event.frontier,
                expected_frontier=self.emit_frontier,
            )
        self.emitted.append(event)
        self.emit_frontier += 1

    def flush_oldest(self, internal: bool = False) -> None:
        self._flush(0, "flush_oldest", internal)

    def flush_newest(self) -> None:
        self._flush(-1, "flush_newest")

    def publish_cache(self) -> None:
        self._step("publish_cache")
        if self.block_key is None or self.materialized_tokens is None:
            return
        self.protocol_checks += int(self.policy in (Policy.EPOCH_ONLY, Policy.VERSIONED))
        expected_key = token_digest(self.materialized_tokens)
        if self.block_key != expected_key:
            self._record(
                "cache_key_payload_mismatch",
                key=self.block_key,
                expected_key=expected_key,
            )
        if self.materialized_revision != self.revision:
            self._record(
                "stale_materialized_state_published",
                materialized_revision=self.materialized_revision,
            )

    def preempt(self) -> None:
        self._step("preempt")
        self.materialized_tokens = None
        self.materialized_revision = None
        self.preempted = True

    def fold_after_preempt(self) -> None:
        self._step("fold_after_preempt")
        if not self.preempted:
            return
        if self.policy in (Policy.MUTABLE, Policy.HASH_ONLY, Policy.EPOCH_ONLY):
            before = self.tokens
            self.tokens = self.materialized_tokens or ()
            if self.tokens != before:
                self._record(
                    "logical_history_lost",
                    tokens_before=len(before),
                    tokens_after=len(self.tokens),
                )
        self.preempted = False

    def enabled_actions(self, max_outstanding: int = 2) -> tuple[str, ...]:
        actions = ["append_input"]
        if self.tokens:
            actions.append("replace_last")
        if len(self.queue) < max_outstanding:
            actions.append("schedule")
        if self.queue:
            actions.append("complete_oldest")
        if len(self.queue) > 1:
            actions.append("complete_newest")
        if self.pending:
            actions.append("flush_oldest")
        if len(self.pending) > 1:
            actions.append("flush_newest")
        if self.materialized_tokens is not None:
            actions.extend(("publish_cache", "preempt"))
        if self.preempted:
            actions.append("fold_after_preempt")
        return tuple(actions)

    def apply(self, action: str) -> None:
        operation = getattr(self, action, None)
        if operation is None or action.startswith("_"):
            raise ValueError(f"unknown action: {action}")
        operation()

    def signature(self) -> tuple[Any, ...]:
        return (
            self.tokens,
            self.revision,
            self.incarnation,
            self.frontier,
            self.schedule_frontier,
            self.emit_frontier,
            self.block_key,
            self.materialized_tokens,
            self.materialized_revision,
            self.preempted,
            tuple(self.queue),
            tuple(self.pending),
            tuple(self.emitted),
            tuple(v["kind"] for v in self.violations),
        )

    def stats(self) -> dict[str, Any]:
        kinds = sorted({violation["kind"] for violation in self.violations})
        return {
            "policy": self.policy.value,
            "session_id": self.session_id,
            "operations": self.operation_count,
            "protocol_checks": self.protocol_checks,
            "violations": len(self.violations),
            "violation_types": kinds,
            "discarded_work": self.discarded_work,
            "discarded_events": self.discarded_events,
            "rebuild_tokens": self.rebuild_tokens,
            "computed_tokens": self.computed_tokens,
            "drained_work": self.drained_work,
            "missing_callbacks": self.missing_callbacks,
            "updates": self.update_count,
        }
