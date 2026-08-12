#!/usr/bin/env python3
"""Plot the four iteration-level loss curves produced by LossTraceRecorder."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_path', type=Path,
                        help='Run directory or path to iteration_curves.csv')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output stem or .png/.pdf path; both PNG and PDF are always written')
    parser.add_argument('--smooth-window', type=int, default=1,
                        help='Centered rolling-mean window in iterations; 1 disables smoothing')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--title', type=str, default='Iteration-level loss by sample group')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smooth_window <= 0:
        raise ValueError('--smooth-window must be greater than zero')
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            'Plotting requires pandas and matplotlib. Install the project optional dependencies '
            'or run: pip install pandas matplotlib'
        ) from exc

    curve_path = args.trace_path
    if curve_path.is_dir():
        curve_path = curve_path / 'iteration_curves.csv'
    if not curve_path.exists():
        raise FileNotFoundError(curve_path)

    frame = pd.read_csv(curve_path).sort_values(['task_id', 'task_iteration_id'])
    output_path = args.output or curve_path.parent / 'four_loss_curves'
    output_stem = output_path.with_suffix('') if output_path.suffix.lower() in {'.png', '.pdf'} else output_path
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': args.dpi,
        'savefig.bbox': 'tight',
    })

    styles = {
        'noisy_loss': ('Noisy data', '#D55E00', '-'),
        'hard_old_loss': ('Hard old data', '#E69F00', '--'),
        'easy_old_loss': ('Easy old data', '#009E73', '-.'),
        'new_loss': ('New data', '#0072B2', ':'),
    }
    fig, ax = plt.subplots(figsize=(6.75, 3.4))
    for column, (label, color, linestyle) in styles.items():
        values = frame[column]
        if args.smooth_window > 1:
            values = values.rolling(args.smooth_window, center=True, min_periods=1).mean()
        valid = values.notna()
        ax.plot(frame.loc[valid, 'x_epoch'], values[valid], label=label, color=color,
                linestyle=linestyle, linewidth=1.7)

    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Observed-label CE loss')
    ax.set_title(args.title)
    ax.grid(True, linestyle='-', alpha=0.15)
    ax.legend(ncol=2)
    fig.tight_layout()
    png_path = output_stem.with_suffix('.png')
    pdf_path = output_stem.with_suffix('.pdf')
    fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(png_path.resolve())
    print(pdf_path.resolve())


if __name__ == '__main__':
    main()
