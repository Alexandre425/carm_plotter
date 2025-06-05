import math

from matplotlib import ticker

def get_base10_prefix(x: float) -> str:
    prefixes = [' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

    exp = min(int(math.log10(abs(x)) / 3), len(prefixes) - 1)
    return prefixes[exp]

def get_base10_prefix_scale(x: float) -> int:
    exp = int(math.log10(abs(x)) / 3)
    return 10**(3*exp)


def with_base10_prefix(x: float, decimal_places: int = 0) -> str:
    prefixes = [' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

    if x == 0:
        return "0"

    exp = min(int(math.log10(abs(x)) / 3), len(prefixes) - 1)
    val = f"{x / 10**(3*exp):.{decimal_places}f}".rstrip("0").rstrip(".")

    return f"{val}{prefixes[exp]}"


def with_base2_prefix(x: float, decimal_places: int = 0) -> str:
    prefixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi']

    if x == 0:
        return "0"

    exp = int(math.log2(abs(x)) / 10)
    val = f"{x / 2**(10*exp):.{decimal_places}f}".rstrip("0").rstrip(".")
    prefix = prefixes[exp] if exp >= 0 else ''

    # use 2^N if we can't represent it in the necessary decimal places
    log = int(math.log2(abs(x)))
    if -log > decimal_places:
        return f"$2^{{{log}}}$"
    else:
        return f"{val}{prefix}"


def tick_formatter_base2(val, _pos):
    "Tick formatter for the x axis, applies a base 2 prefix (e.g. Ki, Mi)"
    return with_base2_prefix(val, decimal_places=3)


def tick_formatter_base10(val, _pos):
    "Tick formatter for the y axis, applies a base 10 prefix (e.g. M, G)"
    return with_base10_prefix(val, decimal_places=2)


class ScaledTickFormatter:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, val, pos):
        return with_base2_prefix(val / self.scale, decimal_places=3)


class ScaledTickLocator(ticker.Locator):
    def __init__(self, scale: float, max_ticks: int = 4):
        self.scale = scale
        self.max_ticks = max_ticks

    def tick_values(self, vmin, vmax):
        exp_max = int(math.log2(vmax / self.scale))
        exp_min = int(math.log2(vmin / self.scale))

        step = 1 + (exp_max - exp_min) // self.max_ticks
        ticks = [self.scale * 2 ** (exp) for exp in range(exp_max, exp_min, -step)]

        # Fall back to default locator if fewer than 2 ticks
        if len(ticks) < 2:
            return ticker.MaxNLocator(nbins=self.max_ticks).tick_values(vmin, vmax)

        return ticks

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)