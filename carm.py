"Definitions of classes pertaining to the CARM"

import json

from .num_formatting import with_base10_prefix

class CARMData:
    def __init__(self, memory_bandwidth: "list[float]", peak_performance: float, frequency_hz: float) -> None:
        self.memory_bandwidth = memory_bandwidth
        self.peak_performance = peak_performance
        self.frequency        = frequency_hz
        self.ridge_points     = [peak_performance / bw for bw in memory_bandwidth]

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> "dict":
        return {
            "memory_bandwidth": self.memory_bandwidth,
            "peak_performance": self.peak_performance,
            "frequency":        self.frequency,
        }

    def to_file(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def from_dict(d: dict) -> "CARMData":
        return CARMData(d["memory_bandwidth"], d["peak_performance"], d["frequency"])


class CARMPoint:
    def __init__(self, cycles: int, bytes: int, flops: int, frequency_hz: float) -> None:
        self.cycles               = cycles
        self.bytes                = bytes
        self.flops                = flops
        self.frequency            = frequency_hz
        self.arithmetic_intensity = flops / bytes
        self.performance          = frequency_hz * flops / cycles

    def __str__(self) -> str:
        return f"{with_base10_prefix(self.performance, decimal_places=3)}FLOP/s @ AI {self.arithmetic_intensity:.3f}"

    def to_dict(self) -> "dict":
        return {
            "cycles":    self.cycles,
            "bytes":     self.bytes,
            "flops":     self.flops,
            "frequency": self.frequency,
        }

    def from_dict(d: dict, frequency_hz: float) -> "CARMPoint":
        flops = d.get("flops", d.get("ops"))
        if not flops:
            raise KeyError("flops|ops")
        return CARMPoint(d["cycles"], d["bytes"], flops, frequency_hz)