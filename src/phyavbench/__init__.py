from __future__ import annotations

from phyavbench.cli import cli as _cli


def cli() -> int:
    return _cli()


def main() -> int:
    return cli()
