"""Launch independent Linear Oracle SAP runs that differ only in alpha."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Sequence


DEFAULT_SCALES = (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)


def _without_existing_scale(command: Sequence[str]) -> list[str]:
    cleaned = []
    skip_next = False
    for argument in command:
        if skip_next:
            skip_next = False
            continue
        if argument == '--sap_oracle_scale':
            skip_next = True
            continue
        if argument.startswith('--sap_oracle_scale='):
            continue
        cleaned.append(argument)
    if skip_next:
        raise ValueError('--sap_oracle_scale requires a value')
    return cleaned


def build_sweep_commands(
    base_command: Sequence[str],
    scales: Sequence[float],
) -> list[list[str]]:
    """Return one independent command per alpha, preserving every other arg."""
    if not base_command:
        raise ValueError('base training command must not be empty')
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError('all alpha values must be positive')
    cleaned = _without_existing_scale(base_command)
    return [
        [*cleaned, '--sap_oracle_scale', f'{float(scale):g}']
        for scale in scales
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run independent Linear Oracle SAP experiments for each alpha.',
    )
    parser.add_argument('--scales', nargs='+', type=float, default=DEFAULT_SCALES)
    parser.add_argument(
        'command', nargs=argparse.REMAINDER,
        help='Training command after `--`, for example: -- python main.py ...',
    )
    args = parser.parse_args()
    base_command = args.command[1:] if args.command[:1] == ['--'] else args.command

    for command in build_sweep_commands(base_command, args.scales):
        print(f'Running: {shlex.join(command)}', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
