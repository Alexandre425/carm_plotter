import math


def with_base10_prefix(x: float, decimal_places: int = 0) -> str:
    prefixes = [' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

    if x == 0:
        return "0"

    exp = min(int(math.log10(abs(x)) / 3), len(prefixes) - 1)
    val = f"{x / 10**(3*exp):.{decimal_places}f}".strip("0").strip(".")

    return f"{val}{prefixes[exp]}"


def with_base2_prefix(x: float, decimal_places: int = 0) -> str:
    prefixes = [' ', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi']

    if x == 0:
        return "0"

    exp = min(int(math.log2(abs(x)) / 10), len(prefixes) - 1)
    val = f"{x / 2**(10*exp):.{decimal_places}f}".strip("0").strip(".")
    return f"{val}{prefixes[exp]}"


def tick_formatter_base2(val, _pos):
    "Tick formatter for the x axis, applies a base 2 prefix (e.g. Ki, Mi)"
    return with_base2_prefix(val, decimal_places=0)


def tick_formatter_base10(val, _pos):
    "Tick formatter for the y axis, applies a base 10 prefix (e.g. M, G)"
    return with_base10_prefix(val, decimal_places=2)