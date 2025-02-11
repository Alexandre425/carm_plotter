"Definitions of classes pertaining to the CARM"

class CARMData:
    def __init__(self, memory_bandwidth: "list[float]", peak_performance: float, frequency: float, color: bool = True) -> None:
        self.memory_bandwidth = memory_bandwidth
        self.peak_performance = peak_performance
        self.ridge_points     = [peak_performance / bw for bw in memory_bandwidth]
        self.frequency        = frequency
        #self.color            = color


    def to_dict(self) -> "dict":
        return {
            "memory_bandwidth": self.memory_bandwidth,
            "peak_performance": self.peak_performance,
            "frequency": self.frequency,
        }

    def from_dict(d: dict) -> "CARMData":
        return CARMData(d["mem_bw"], d["peak_perf"], d["frequency"], d["color"])


class CARMPoint:
    def __init__(self, cycles: int, bytes: int, flops: int, frequency: float, tech_nodes: list, power: list, energy: list) -> None:
        self.arithmetic_intensity = flops / bytes
        self.performance          = frequency * flops / cycles
        # Keep the cycles and bytes for the weighted addition of phases
        self.cycles               = cycles
        self.bytes                = bytes
        # Power and energy obtained with MCPAT
        self.tech_nodes           = tech_nodes
        self.power                = power
        self.energy               = energy

    def __str__(self) -> str:
        return f"{self.performance} GFLOP/s @ AI {self.arithmetic_intensity}"

    def from_dict(d: dict, frequency: float) -> "CARMPoint":
        return CARMPoint(d["cycles"], d["bytes"], d["flops"], frequency, [], [], [])