#!/usr/bin/env python3

import os
import sys
import json
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
import argparse

from .carm import CARMData, CARMPoint
from .num_formatting import (
    tick_formatter_base2, get_base10_prefix, get_base10_prefix_scale,
    ScaledTickFormatter, ScaledTickLocator
)


def get_mem_level_names(num_levels):
    """Generates names like [L1, L2, ..., LN, DRAM], for plotting purposes"""
    return [f"L{i+1}" for i in range(num_levels - 1)] + ["DRAM"]
    

def plot_rooflines(carm: CARMData, roof_names: "list[str]" = None, label_roofs: bool = True, 
                   color: str = None, axis_labels: bool = True, linewidth: float = None, 
                   label_override: str = None) -> None:
    """Plots the CARM from the memory bandwidth and peak performance

    Args:
        carm (CARMData): The roofline data to plot
        roof_names (list[str], optional): The list of names of the roofs, starting from the highest bandwidth. Defaults to None.
        label_roofs (bool, optional): Apply roof labels. Defaults to True.
        color (str, optional): The color of the roofs. All roofs will share the same color if provided. 
            Defaults to None, which automatically colors the roofs.
        axis_labels (bool, optional): Apply axis labels (AI, performance). Defaults to True.
        linewidth (float, optional): Width of the roof lines. Defaults to None.
        label_override (str, optional): Single label that overrides the roof names. Applies to all roofs. Defaults to None.
    """

    num_rooflines = len(carm.ridge_points)
    if roof_names is None:
        roof_names = get_mem_level_names(num_rooflines)
    applied_overrided_label = False

    plot_max_y = carm.peak_performance * 2
    numerical_prefix = get_base10_prefix(plot_max_y)
    performance_scale = get_base10_prefix_scale(plot_max_y)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][3-num_rooflines:]
    for ridge, bw, roof_name, col in zip(carm.ridge_points, carm.memory_bandwidth, roof_names, colors):
        # plot the roofline way to the left and right of the ridge point
        x_lim = (ridge / 2**20, ridge * 2**20)
        x = [x_lim[0], ridge, x_lim[1]]
        y = [bw * x_lim[0], carm.peak_performance, carm.peak_performance]
        if color is not None:
            roof_name = None
            col = color if color is not None else "grey"

        if label_override:
            if not applied_overrided_label:
                roof_name = label_override
            applied_overrided_label = True
        elif not label_roofs:
            roof_name = None

        plt.plot(x, y, color=col, label=roof_name, linewidth=linewidth)

    plt.grid(True)
    if axis_labels:
        plt.xlabel("Arithmetic Intensity [FLOP/Byte]")
        plt.ylabel(f"Performance\n[{numerical_prefix}FLOP/s]")
    plt.xscale("log", base = 2)
    plt.yscale("log", base = 2)

    axes: plt.Axes = plt.gca()
    axes.xaxis.set_major_formatter(tick_formatter_base2)
    axes.yaxis.set_major_locator(ScaledTickLocator(performance_scale))
    axes.yaxis.set_major_formatter(ScaledTickFormatter(performance_scale))

    # Center the first ridge point
    max_x = carm.ridge_points[-1] * 2
    center_distance_to_right = max_x / carm.ridge_points[0]
    min_x = carm.ridge_points[0] / center_distance_to_right
    plt.xlim(min_x, max_x)
    # Scale y so the DRAM roofline intersects the lower left corner
    min_y = min_x * carm.memory_bandwidth[-1]
    plt.ylim(min_y, carm.peak_performance * 2)


def plot_points(points: "dict[str, CARMPoint]"):
    """Plots `CARMPoint`s according to their AI and performance

    Args:
        points (dict[str, CARMPoint]): Dict with the point names (used as the label) and respective position
    """
    arithmetic_intensity = [p.arithmetic_intensity for _, p in points.items()]
    performance = [p.performance for _, p in points.items()]
    name = [name for name, p in points.items()]

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


def plot_grouped_points(points: "dict[str, list[CARMPoint]]"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", action="store_true", help="Plot points")
    parser.add_argument("-r", action="store_true", help="Plot the AI range")
    parser.add_argument("-f", action="store_true", help="The file containing ")
    args = parser.parse_args()
