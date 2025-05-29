import math
import json
import argparse
import matplotlib.pyplot as plt

from .carm import CARMData
from .num_formatting import (
    get_base10_prefix, get_base10_prefix_scale, with_base10_prefix, tick_formatter_base2,
    tick_formatter_base10, ScaledTickFormatter, ScaledTickLocator
)


def get_bandwidth(memory_benchmark: "dict[str, int]", frequency_hz: int, plot: bool) -> "list[float]":
    """Identifies and returns the memory bandwidth of each cache level from the memory benchmark"""

    bytes = [int(b) for b in memory_benchmark.keys()]
    cycles = [c for c in memory_benchmark.values()]
    bandwidth = [frequency_hz * (b / c) for b, c in zip(bytes, cycles)]

    CLUSTER_THRESHOLD = 0.4

    clusters = [[]]
    for bandwidth_point in zip(bytes, bandwidth):
        current_cluster = clusters[-1]
        if len(current_cluster) == 0:
            clusters[-1].append(bandwidth_point)
            continue

        cluster_avg = sum(c[1] for c in clusters[-1]) / len(current_cluster)

        # if the point is close to the cluster average, add it, otherwise create a new cluster
        if abs(bandwidth_point[1] - cluster_avg) < CLUSTER_THRESHOLD * cluster_avg:
            clusters[-1].append(bandwidth_point) # add to the last cluster
        else:
            # overwrite the last cluster if it's too small, or create a new one if it's large enough
            if len(current_cluster) < 3:
                clusters[-1] = [bandwidth_point]
            else:
                clusters.append([bandwidth_point])

    # clean up the clusters, remove outliers
    for cluster_idx, cluster in enumerate(clusters):
        # remove the 50% top outliers (minimum of 1, maximum of total-1)
        points_to_remove = max(1, min(int(len(cluster) * 0.5), len(cluster)-1))
        for _ in range(points_to_remove):
            average = sum(p[1] for p in cluster) / len(cluster)
            # For the first cluster (L1), bias the deviation to keep the top points
            if cluster_idx == 0:
                deviation = [abs(average - p[1]) + max(0, average - p[1]) for p in cluster]
            else:
                deviation = [abs(p[1] - average) for p in cluster]
            max_dev = -math.inf
            max_idx = None
            for dev, idx in zip(deviation, range(len(deviation))):
                if dev > max_dev:
                    max_dev = dev
                    max_idx = idx
            cluster.pop(max_idx)

    level_bandwidth = [sum(p[1] for p in c) / len(c) for c in clusters]
    #level_bandwidth = [max(p[1] for p in c) for c in clusters]

    # plot the bandwidth and clusters if requested
    if plot:
        plot_max_y = max(bandwidth)
        numerical_prefix = get_base10_prefix(plot_max_y)
        performance_scale = get_base10_prefix_scale(plot_max_y)
        ax = plt.subplot(1, 2, 1)
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xlabel("Data Traffic [Bytes]")
        plt.ylabel(f"Memory Bandwidth [{numerical_prefix}B/s]")

        plt.grid(True)
        ax.xaxis.set_major_formatter(tick_formatter_base2)
        ax.yaxis.set_major_locator(ScaledTickLocator(performance_scale))
        ax.yaxis.set_major_formatter(ScaledTickFormatter(performance_scale))

        # plot microbenchmark results
        plt.plot(bytes, bandwidth, marker='x', c='g')
        # identify clusters and plot bandwidth line
        for cluster, bandwidth in zip(clusters, level_bandwidth):
            x = [c[0] for c in cluster]
            y = [c[1] for c in cluster]
            plt.plot(x, y, marker='o', c='r')
            plt.axhline(bandwidth, ls=':', c='b')
            # annotate with bandwidth
            plt.annotate(with_base10_prefix(bandwidth, decimal_places=3), c='b',
                         xy=(bytes[0], bandwidth), xytext=(0, 0.2), textcoords='offset fontsize')

    return level_bandwidth


def get_peak_performance(arithmetic_benchmark: "dict[str, int]", frequency_hz: int, plot: bool) -> float:
    """Returns the peak arithmetic performance from the arithmetic benchmark"""

    arith_ops = [int(o) for o in arithmetic_benchmark.keys()]
    cycles = [c for c in arithmetic_benchmark.values()]
    performance = [frequency_hz * (o / c) for o, c in zip(arith_ops, cycles)]

    peak_perf = max(performance)

    if plot:

        plot_max_y = max(performance)
        numerical_prefix = get_base10_prefix(plot_max_y)
        performance_scale = get_base10_prefix_scale(plot_max_y)

        ax = plt.subplot(1, 2, 2)
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xlabel("Arithmetic Operations [Ops]")
        plt.ylabel(f"Arithmetic Performance [{numerical_prefix}Ops/s]")

        plt.grid(True)

        ax.xaxis.set_major_formatter(tick_formatter_base2)
        ax.yaxis.set_major_locator(ScaledTickLocator(performance_scale))
        ax.yaxis.set_major_formatter(ScaledTickFormatter(performance_scale))

        plt.plot(arith_ops, performance, marker='x', c='g')
        plt.axhline(peak_perf, ls=':', c='b')
        plt.annotate(with_base10_prefix(peak_perf, decimal_places=3), c='b',
                     xy=(arith_ops[0], peak_perf), xytext=(0, 0.2), textcoords='offset fontsize')

    return max(performance)


def build_carm(benchmark_results: "dict[str, dict[str, int]]", frequency_hz: int, plot_path: str = None) -> CARMData:
    """Builds a CARM model from benchmark results, optionally plotting the memory bandwidth and peak performance

    Args:
        benchmark_results (dict[str, dict[str, int]]): The CARM microbenchmark results
        frequency_hz (int): The frequency of the core (shared by the CSRs that measure cycles)
        plot_path (str, optional): The path to the bandwidth and peak performance plot. Won't generate if `None` is passed

    Returns:
        CARMData: The built CARM model
    """

    plot = plot_path is not None

    if plot:
        plt.figure(figsize=(14, 6))
        plt.tight_layout()

    level_bandwidth = get_bandwidth(benchmark_results["memory"], frequency_hz, plot)
    arithmetic_perf = get_peak_performance(benchmark_results["arithmetic"], frequency_hz, plot)

    carm = CARMData(level_bandwidth, arithmetic_perf, frequency_hz)

    if plot:
        plt.savefig(f"{plot_path}", bbox_inches='tight')
        plt.clf()

    return carm


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CARM Builder", description="Tool to build the CARM from benchmark results")
    parser.add_argument("input", help="Path to the json file containing benchmark results")
    parser.add_argument("frequency", type=int, help="Frequency of the core in Hz")
    parser.add_argument("--output", "-o", help="Destination path for the json file containing the CARM data, outputs to stdout if omitted")
    parser.add_argument("--plot", "-p", help="Destination path for the memory and arithmetic plot", metavar="PLOT_PATH")
    args = parser.parse_args()

    with open(f"{args.input}", "r") as file:
        benchmark_results = json.load(file)

    carm = build_carm(benchmark_results, args.frequency, plot_path=args.plot)

    if args.output:
        carm.to_file(args.output)
    else:
        print(carm)
