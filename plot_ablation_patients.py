#!/usr/bin/env python3.7

import argparse
from pathlib import Path
from typing import Dict, List
from operator import itemgetter
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import map_


def get_curve(p: Path) -> str:
    return '_'.join(p.parents[0].name.split("_")[:-1])


def get_number(p: Path) -> int:
    return int(p.parents[0].name.split('_')[-1])


def main(args: argparse.Namespace) -> None:
    plt.rc('font', size=args.fontsize)

    paths: List[Path] = map_(Path, args.metric_logs)

    numbers: List[int] = map_(get_number, paths)
    curves: List[str] = map_(get_curve, paths)
    uniq_curves: List[str] = sorted(set(curves))
    if args.labels:
        assert len(uniq_curves) == len(args.labels)

    print(args.metric_logs)
    print(paths)
    print(numbers)
    print(curves)

    idx: List[int] = sorted(set(numbers))

    fig = plt.figure(figsize=args.figsize)
    ax = fig.gca()

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 76])

    ax.set_yticks(np.mgrid[0:1.1:.2])
    ax.set_xticks([0] + idx)

    ax.set_xlabel("# of fully anotated patient")
    ax.set_ylabel("DSC")
    ax.grid(True, axis='y')

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"Comparison between {', '.join(uniq_curves)} training")

    numbers_max: Dict[int, float] = defaultdict(lambda: 0)
    for i, curve in enumerate(uniq_curves):
        curve_xs: List[int] = []
        curve_ys: List[float] = []
        curve_ys_std: List[float] = []

        for sub_curve, number, path in zip(curves, numbers, paths):
            if sub_curve != curve:
                continue

            curve_xs.append(number)

            arr: np.ndarray = np.load(path)
            arr_mean: np.ndarray = arr.mean(axis=1)
            val: float
            if args.mean_last:
                curve_vals: np.ndarray = arr_mean[-args.last_epc:, args.metric_axis]
                val = curve_vals.mean()
                std: float = curve_vals.std()

                curve_ys_std.append(std)
            else:
                val = arr_mean[:, args.metric_axis].max()

            curve_ys.append(val)

            numbers_max[number] = max(val, numbers_max[number])

        # Zip tuple, sort it according to item 0, unzip
        curve_xs, curve_ys = zip(*sorted(zip(curve_xs, curve_ys), key=itemgetter(0)))

        if args.mean_last:
            curve_xs, curve_ys_std = zip(*sorted(zip(curve_xs, curve_ys_std), key=itemgetter(0)))

            lab: str = args.labels[i] if args.labels else curve
            plt.errorbar(curve_xs, curve_ys, yerr=curve_ys_std, capsize=3, label=lab)
        else:
            plt.plot(curve_xs, curve_ys, label=curve)

    for x, y in numbers_max.items():
        plt.plot([x, x], [0, y], color='gray', linewidth=.25)
    ax.legend(loc=args.loc)

    fig.tight_layout()
    if args.savefig:
        fig.savefig(args.savefig)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument("--metric_logs", type=str, nargs="+", required=True)
    parser.add_argument("--metric_axis", type=int, required=True)

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mean_last", action="store_true")

    parser.add_argument("--title", type=str, default='')
    parser.add_argument("--savefig", type=str, default=None)
    parser.add_argument("--figsize", type=int, nargs='*', default=[14, 9])
    parser.add_argument("--fontsize", type=int, default=10)
    parser.add_argument("--labels", type=str, nargs='*')
    parser.add_argument("--last_epc", type=int, default=20)
    parser.add_argument("--loc", type=str, default=None, choices=matplotlib.legend.Legend.codes.copy())

    args = parser.parse_args()
    print("\n", args)

    return args


if __name__ == "__main__":
    main(get_args())
