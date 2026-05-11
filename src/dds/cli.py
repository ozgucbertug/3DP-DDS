"""Shared tyro-backed helpers for repository CLI entry points."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import tyro

T = TypeVar("T")


def parse_cli(config_type: type[T], argv: Sequence[str] | None = None) -> T:
    """Parse a typed CLI configuration with tyro."""

    args = list(argv) if argv is not None else None
    return tyro.cli(config_type, args=args)


def run_cli(
    config_type: type[T],
    handler: Callable[[T], None],
    argv: Sequence[str] | None = None,
) -> None:
    """Parse a typed CLI configuration and dispatch to a handler."""

    handler(parse_cli(config_type, argv=argv))
