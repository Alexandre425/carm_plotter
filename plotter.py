#!/usr/bin/env python3

import os
import sys
import json
import math
import matplotlib.pyplot as plt
import argparse

from .carm import CARMData, CARMPoint
from .num_formatting import with_base2_prefix, with_base10_prefix

def round_to_pow2(num) -> int:
    return int(2 ** round(math.log2))


def get_mem_level_names(num_levels):
    """Generates names like [L1, L2, ..., LN, DRAM], for plotting purposes"""
    if num_levels < 3:
        return ["L2", "DRAM"][2-num_levels:]
    else:
        return ["L1V"] + [f"L{i+2}" for i in range(num_levels - 2)] + ["DRAM"]


def convert_plot_ticks(x_ticks: bool = True, y_ticks: bool = True, x_base = 2, y_base = 2):
    get_labels_b2 = lambda x: [with_base2_prefix(l) for l in x]
    get_labels_b10 = lambda x: [with_base10_prefix(l) for l in x]

    xlim, ylim = plt.xlim(), plt.ylim() # keep the same plot limits

    if x_ticks:
        loc, _ = plt.xticks()
        labels = get_labels_b2(loc) if x_base == 2 else get_labels_b10(loc)
        plt.xticks(loc, labels)
    if y_ticks:
        bot, top = plt.ylim()
        # get the prefix-quantity at the top, i.g. 10_000 is 1000 (kilo), 100_000_000 is 1_000_000 (mega)
        top_prefix = 10 ** (3 * math.floor(math.log10(top)/3))
        # get the power of 2 below the multiple of the prefix quant, i.g. 20_000 -> 20 -> 16
        top_p2 = 2 ** int(math.log2(top / top_prefix))
        # generate ticks by halving until we reach the plot bottom
        ticks = []
        val = top_p2 * top_prefix
        while val > bot:
            ticks.append(val)
            val = val // 2

        loc, _ = plt.yticks()
        labels = get_labels_b2(ticks) if y_base == 2 else get_labels_b10(ticks)
        plt.yticks(ticks, labels)

    plt.xlim(xlim), plt.ylim(ylim)


def convert_plot_labels(x_ticks: bool = True, y_ticks: bool = True, x_base = 2, y_base = 2):
    """Converts the current plot's labels to power of 2 notation (kilo, mega, etc)"""

    get_labels_b2 = lambda x: [with_base2_prefix(l) for l in x]
    get_labels_b10 = lambda x: [with_base10_prefix(l) for l in x]

    xlim, ylim = plt.xlim(), plt.ylim()

    if x_ticks:
        loc, _ = plt.xticks()
        labels = get_labels_b2(loc) if x_base == 2 else get_labels_b10(loc)
        plt.xticks(loc, labels)
    if y_ticks:
        loc, _ = plt.yticks()
        labels = get_labels_b2(loc) if y_base == 2 else get_labels_b10(loc)
        plt.yticks(loc, labels)

    plt.xlim(xlim), plt.ylim(ylim)



def plot_rooflines(carm: CARMData, apply_label: bool = True, color_val: str = None, axis_labels: bool = True,
              linewidth: float = None, label_override: str = None) -> None:
    """Plots the CARM from the memory bandwidth and peak fp performance"""

    roof_names = get_mem_level_names(len(carm.memory_bandwidth))
    num_rooflines = len(carm.ridge_points)
    applied_overrided_label = False

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][2-num_rooflines:]#[::-1] # This takes the first N colors and inverts the order
    for ridge, bw, roof_name, col in zip(carm.ridge_points, carm.memory_bandwidth, roof_names, colors):
        # plot the roofline waaay to the left and right of the ridge point
        x_lim = (ridge / 2**20, ridge * 2**20)
        x = [x_lim[0], ridge, x_lim[1]]
        y = [bw * x_lim[0], carm.peak_performance, carm.peak_performance]
        zorder = 2.1
        if not carm.color or color_val is not None:
            roof_name = None
            col = color_val if color_val is not None else "grey"
            zorder = 2

        if label_override:
            if not applied_overrided_label:
                roof_name = label_override
            applied_overrided_label = True
        elif not apply_label:
            roof_name = None

        plt.plot(x, y, color=col, label=roof_name, zorder=zorder, linewidth=linewidth)


    plt.grid(True)
    if axis_labels:
        plt.xlabel("Arithmetic Intensity [FLOP/Byte]")
        plt.ylabel("Performance\n[GFLOP/s]")
    plt.xscale("log", base = 2)
    plt.yscale("log", base = 10)

    convert_plot_labels(x_ticks=False)
    # Center the first ridge point
    max_x = carm.ridge_points[-1] * 2
    center_distance_to_right = max_x / carm.ridge_points[0]
    min_x = carm.ridge_points[0] / center_distance_to_right
    plt.xlim(min_x, max_x)
    # Scale y so the DRAM roofline intersects the lower left corner
    min_y = min_x * carm.memory_bandwidth[-1]
    plt.ylim(min_y, carm.peak_performance * 2)


def plot_points(points: "dict[str, CARMPoint] | dict[str, list[CARMPoint]]"):
    try:
        # try parsing the points as if points: dict[str, CARMPoint]
        arithmetic_intensity = [p.arithmetic_intensity for _, p in points.items()]
        performance = [p.performance for _, p in points.items()]
        name = [name for name, p in points.items()]
    except:
        # if it fails, parse them as a group
        plot_points_grouped(points)
        return

    for ai, perf, name in zip(arithmetic_intensity, performance, name):
        #result: dict = list(result.values())[0]
        if name[0] == "_":
            plt.scatter(ai, perf, label=name, color="w")
            plt.text(ai, perf, str(name[1:]), ha="center", va="center", size=12, color="0.5", weight="bold")
        else:
            plt.scatter(ai, perf, marker='x', label=name, zorder=2.04)
        plt.legend()

    xlim = plt.xlim()
    plt.xlim(min(xlim[0], min(arithmetic_intensity)/2), max(xlim[1], max(arithmetic_intensity)*2))
    ylim = plt.ylim()
    plt.ylim(min(ylim[0], min(performance)/2), max(ylim[1], max(performance)*2))


def plot_points_grouped(points: "dict[str, list[CARMPoint]]"):
    colors = ["grey", "red"]
    coli = 0
    for group_name, group_points in points.items():
        arithmetic_intensity = [p.arithmetic_intensity for p in group_points]
        performance = [p.performance for p in group_points]
        plt.scatter(arithmetic_intensity, performance, marker='x',
                    label=group_name, color=colors[coli], zorder=2.04)
        coli += 1

    plt.legend()
    xlim = plt.xlim()
    plt.xlim(min(xlim[0], min(arithmetic_intensity)/2), max(xlim[1], max(arithmetic_intensity)*2))


def zoom_on_points(points: "dict[str, CARMPoint] | dict[str, list[CARMPoint]]"):
    try:
        arithmetic_intensity = [p.arithmetic_intensity for _, p in points.items()]
        performance = [p.performance for _, p in points.items()]
    except:
        grouped = []
        [p for p in points.items()]

    plt.xlim(min(arithmetic_intensity) / 2, max(arithmetic_intensity) * 2)
    #plt.ylim(bottom=min(performance) / 2)


def zoom_on_points(points: "dict[str, CARMPoint] | dict[str, list[CARMPoint]]"):
    try:
        arithmetic_intensity = [p.arithmetic_intensity for _, p in points.items()]
        performance = [p.performance for _, p in points.items()]
    except:
        grouped = []
        [p for p in points.items()]

    plt.xlim(min(arithmetic_intensity) / 2, max(arithmetic_intensity) * 2)
    #plt.ylim(bottom=min(performance) / 2)


def carm_plot_lims(carms: "list[CARMData]"):
    """Adjusts the plotting limits based off of a list of CARMs and points, ensuring they are all captured in the plot"""
    performance = [p.peak_performance for p in carms]
    min_peak_perf = min(performance)
    max_peak_perf = max(performance)
    bandwidth = [bw for p in carms for bw in p.memory_bandwidth]
    min_bandwidth = min(bandwidth)
    max_bandwidth = max(bandwidth)
    left_ridge_point = min_peak_perf / max_bandwidth
    right_ridge_point = max_peak_perf / min_bandwidth

    convert_plot_labels(x_ticks=False)
    horizontal_margin = 2
    max_x = right_ridge_point * horizontal_margin
    min_x = left_ridge_point / horizontal_margin
    min_y = min_x * min_bandwidth
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_peak_perf * 2)

def highlight_ai_range(carm: CARMData, ai_range: "tuple[float, float]"):
    # AI range shading
    bw = carm.memory_bandwidth[0]
    pk = carm.peak_performance
    ai_range = (3/32, 1/10)
    #ai_range = (21/4, 51/8) # mandelbrot
    roof = [min(bw * ai, pk) for ai in ai_range]
    plt.fill_between(ai_range, roof, [-1e99, -1e99], color="#FFB2B2", alpha=1, zorder = 2.01)


def plot_carm(_plot_points: bool, _plot_ai_range: bool):
    os.chdir(sys.path[0])
    perf_file   = open("../results/carm_perf.json", 'r')

    # Load and parse system performance into the CARM data
    carm_dicts: "list[dict[str, any]]" = json.load(perf_file)
    perf_file.close()
    all_carms: "list[CARMData]" = []
    for carm_dict in carm_dicts:
        carm = CARMData.from_dict(carm_dict)
        all_carms.append(carm)
    # the "main" model will be the the first found with color (there shouldn't be a need for more than one per plot)
    main_CARM = all_carms[0]
    for c in all_carms:
        if c.color:
            main_CARM = c
            break

    plt.clf()
    plt.figure(figsize=(4,1.4))

    # CARM plots
    for model in all_carms:
        plot_rooflines(model, apply_label=(model==main_CARM))

    carm_plot_lims(all_carms)
    plt.autoscale(False)

    # Performance points
    if _plot_points:
        # Load and parse points aka benchmark performance
        points_file = open("../results/carm_points.json", 'r')
        raw_points: dict = json.load(points_file)
        points_file.close()
        points: "dict[CARMPoint] | dict[str, list[CARMPoint]]" = {}
        for name, result in raw_points.items():
            if isinstance(result, list):
                points[name] = [CARMPoint.from_dict(d, main_CARM.frequency) for d in result]
            elif isinstance(result, dict):
                points[name] = CARMPoint.from_dict(result, main_CARM.frequency)

        plot_points(points)
    if _plot_ai_range:
        pass
        #highlight_ai_range(main_CARM, AI_RANGE)

    if not os.path.exists("../results/"):
        os.mkdir("../results/")

    plt.legend(loc='lower right')#'best')
    plt.savefig("../results/carm.png", bbox_inches="tight", dpi=300)
    plt.savefig("../results/carm.pdf", bbox_inches="tight")
    plt.savefig("../results/carm.svg", bbox_inches="tight")

    if _plot_points:
        zoom_on_points(points)
        plt.savefig("../results/carm_zoomed.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", action="store_true", help="Plot points")
    parser.add_argument("-r", action="store_true", help="Plot the AI range")
    parser.add_argument("-f", action="store_true", help="The file containing ")
    args = parser.parse_args()
    plot_carm(args.p, args.r)
