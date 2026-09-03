"""Small statistics helpers, written out so the reported intervals are inspectable."""

from __future__ import annotations

import math


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Not the normal approximation. With ~150 held-out tickets and a win-rate that may
    land near 0 or 1, the normal approximation produces intervals that run past the
    [0, 1] boundary and understates uncertainty exactly where this benchmark is most
    likely to sit. Wilson stays inside the boundary and is well behaved at small n.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_sided_binomial_p(successes: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test against p0.

    Exact rather than a normal-approximation z-test: at n in the low hundreds with a
    proportion near the boundary, the approximation's p-value is not trustworthy, and
    this benchmark's whole value depends on the honesty of that number.
    """
    if n == 0:
        return 1.0

    def pmf(k: int) -> float:
        return math.comb(n, k) * p0**k * (1 - p0) ** (n - k)

    observed = pmf(successes)
    # Sum every outcome at least as extreme as the observed one, with a relative
    # tolerance so floating-point noise does not exclude the symmetric partner.
    total = sum(pmf(k) for k in range(n + 1) if pmf(k) <= observed * (1 + 1e-9))
    return min(1.0, total)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0
