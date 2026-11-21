"""The seam between an approved move and the platform that makes it.

PuckPilot decides what to propose; a person approves it; an executor carries
it out. This module is the contract between the last two, so the tool's
correctness never depends on how a particular platform is driven:

- `TransactResult` - what every executor reports, the same way: whether the
  final step was taken (`submitted`), whether it worked, and whether the move
  can no longer be made as approved (`final`, so the run cancels it instead of
  retrying it every run).
- `Executor` - `kinds`, the `moves.MoveKind` values it can make, and
  `make(manager, team_key, proposal, dry_run=False)`. An approved move of a
  kind the installed executor cannot make is reported to the person, never
  skipped silently (`transactions.carry_out`).
- `installed()` - the executor this machine has, or None. Two can exist:
  the documented platform API (once its write access is granted), and an
  optional local module a manager installs for their own account. Neither
  ships in this repository; without one, approved moves are the person's to
  make and every card says so.

Adding a platform is a class with `kinds` and `make`; nothing that decides
moves changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from puckpilot.season.moves import MoveKind


@dataclass
class TransactResult:
    ok: bool
    message: str
    lines: list[str] = field(default_factory=list)
    submitted: bool = False  # the final step was taken
    # It can no longer be made as approved (taken, on waivers, a name that does
    # not match): the run cancels it rather than trying again every run.
    final: bool = False


@runtime_checkable
class Executor(Protocol):
    kinds: frozenset[str]

    def make(self, manager, team_key: str, proposal, dry_run: bool = False) -> TransactResult: ...


@dataclass(frozen=True)
class FunctionExecutor:
    """An executor that is a plain `make` function, with the kinds it declares."""

    fn: Callable
    kinds: frozenset[str] = frozenset({MoveKind.ADD_DROP.value})

    def make(self, manager, team_key: str, proposal, dry_run: bool = False) -> TransactResult:
        if dry_run:
            return self.fn(manager, team_key, proposal, dry_run=True)
        return self.fn(manager, team_key, proposal)


def as_executor(obj, kinds: frozenset[str] | None = None) -> Executor | None:
    """`obj` as an `Executor`: one already, or a bare `make` function.

    A function with no declared kinds is taken to make every kind - the case
    for test doubles; an installed module declares its own (`KINDS`).
    """
    if obj is None:
        return None
    if isinstance(obj, Executor):
        return obj
    if callable(obj):
        return FunctionExecutor(obj, kinds or frozenset(k.value for k in MoveKind))
    raise TypeError(f"not an executor: {obj!r}")


def _platform_api() -> Executor | None:
    """The documented platform API's executor, once it can write.

    Not yet: Yahoo's Fantasy API still answers 401 for this application, and
    a write path that cannot be exercised is not shipped. When access is
    granted it lands behind `yahoo/client.py` (the only module allowed to
    import `yahoo_fantasy_api`) and is returned here first.
    """
    return None


def _local() -> Executor | None:
    """The manager's own optional module, if installed (git-ignored)."""
    try:
        from puckpilot.local import transact
    except ImportError:
        return None
    declared = getattr(transact, "KINDS", None)
    kinds = frozenset(declared) if declared else frozenset({MoveKind.ADD_DROP.value})
    return FunctionExecutor(transact.make, kinds)


def installed() -> Executor | None:
    """The executor this machine has: the platform API first, then a local module."""
    return _platform_api() or _local()


def can_make(executor: Executor | None, kind: str) -> bool:
    return executor is not None and kind in executor.kinds
