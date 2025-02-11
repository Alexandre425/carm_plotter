import math
import json
import argparse
import matplotlib.pyplot as plt
import plotter

def get_bandwidth(memory_benchmark: "dict[str, int]", plot: bool) -> "list[float]":
    """Identifies and returns the memory bandwidth of each cache level from the memory benchmark"""

    bytes = [int(b) for b in memory_benchmark.keys()]
    cycles = [c for c in memory_benchmark.values()]
    bandwidth = [b / c for b, c in zip(bytes, cycles)]

    CLUSTER_THRESHOLD = 0.2

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
    for cluster in clusters:
        # remove the 20% top outliers (minimum of 1, maximum of total-1)
        points_to_remove = max(1, min(int(len(cluster) * 0.2), len(cluster)-1))
        for _ in range(points_to_remove):
            average = sum(p[1] for p in cluster) / len(cluster)
            deviation = [abs(p[1] - average) for p in cluster]
            max_dev = -math.inf
            max_idx = None
            for dev, idx in zip(deviation, range(len(deviation))):
                if dev > max_dev:
                    max_dev = dev
                    max_idx = idx
            cluster.pop(max_idx)

    #level_bandwidth = [sum(p[1] for p in c) / len(c) for c in clusters]
    level_bandwidth = [max(p[1] for p in c) for c in clusters]

    # plot the bandwidth and clusters if requested
    if plot:
        plt.subplot(1, 2, 1)
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xlabel("Data Traffic [Bytes]")
        plt.ylabel("Memory Bandwidth [GB/s]")

        # plot microbenchmark results
        plt.plot(bytes, bandwidth, marker='x', c='g')
        # identify clusters and plot bandwidth line
        for cluster, bandwidth in zip(clusters, level_bandwidth):
            x = [c[0] for c in cluster]
            y = [c[1] for c in cluster]
            plt.plot(x, y, marker='o', c='r')
            plt.axhline(bandwidth, ls=':', c='b')

    return level_bandwidth


def get_peak_performance(arithmetic_benchmark: "dict[str, int]", plot: bool) -> float:
    """Returns the peak arithmetic performance from the arithmetic benchmark"""

    arith_ops = [int(o) for o in arithmetic_benchmark.keys()]
    cycles = [c for c in arithmetic_benchmark.values()]
    performance = [o / c for o, c in zip(arith_ops, cycles)]

    peak_perf = max(performance)

    if plot:
        plt.subplot(1, 2, 2)
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xlabel("Arithmetic Operations [Ops]")
        plt.ylabel("Arithmetic Performance [Ops/cycle]")

        plt.plot(arith_ops, performance, marker='x', c='g')
        plt.axhline(peak_perf, ls=':', c='b')


    return max(performance)



def build_carm(benchmark_results: "dict[str, dict[str, int]]", output_path: str = None, plot_path: str = None):
    """Builds the"""

    plot = plot_path is not None

    if plot:
        plt.figure(figsize=(16, 6))

    level_bandwidth = get_bandwidth(benchmark_results["memory"], plot)
    arithmetic_perf = get_peak_performance(benchmark_results["arithmetic"], plot)

    if plot:
        plt.savefig(f"{plot_path}")

    if output_path:
        pass
    else:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CARM Builder", description="Tool to build the CARM from benchmark results")
    parser.add_argument("input", help="Path to the json file containing benchmark results")
    parser.add_argument("--output", "-o", help="Destination path for the json file containing the CARM data, outputs to stdout if omitted")
    parser.add_argument("--plot", "-p", help="Destination path for the memory and arithmetic plot", metavar="PLOT_PATH")
    args = parser.parse_args()

    with open(f"{args.input}", "r") as file:
        benchmark_results = json.load(file)

    build_carm(benchmark_results, args.output, args.plot)
