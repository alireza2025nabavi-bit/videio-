"""Tiny animation toolkit: easings, damped springs and keyed tracks."""
import math


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def seg(t, t0, t1):
    """Normalised progress of t through [t0, t1], clamped to 0..1."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def ease_out_cubic(x):
    x = clamp(x)
    return 1.0 - (1.0 - x) ** 3


def ease_in_cubic(x):
    x = clamp(x)
    return x ** 3


def ease_in_out_cubic(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_quint(x):
    x = clamp(x)
    return 1.0 - (1.0 - x) ** 5


def ease_out_expo(x):
    x = clamp(x)
    return 1.0 if x >= 1.0 else 1.0 - 2.0 ** (-10.0 * x)


def ease_in_expo(x):
    x = clamp(x)
    return 0.0 if x <= 0.0 else 2.0 ** (10.0 * x - 10.0)


def ease_in_quad(x):
    x = clamp(x)
    return x * x


def ease_out_quad(x):
    x = clamp(x)
    return 1 - (1 - x) ** 2


def spring(t, freq=3.0, damp=0.6):
    """Unit step response of a damped spring (0 -> 1, overshoots when damp < 1).

    t    seconds since the spring was released
    freq natural frequency in Hz
    damp damping ratio (1 = critically damped)
    """
    if t <= 0.0:
        return 0.0
    w = 2.0 * math.pi * freq
    if damp < 1.0:
        wd = w * math.sqrt(1.0 - damp * damp)
        return 1.0 - math.exp(-damp * w * t) * (math.cos(wd * t) + (damp * w / wd) * math.sin(wd * t))
    return 1.0 - math.exp(-w * t) * (1.0 + w * t)


class Track:
    """A value that springs / eases from key to key.

    keys: [(time, target, opts)], opts = dict(freq=, damp=) for a spring or
          dict(dur=, ease=callable) for a timed ease.
    """

    def __init__(self, v0, keys=()):
        self.v0 = v0
        self.keys = sorted(keys, key=lambda k: k[0])

    def __call__(self, t):
        v = self.v0
        for tk, target, opts in self.keys:
            if t <= tk:
                break
            if "dur" in opts:
                k = opts.get("ease", ease_out_cubic)(seg(t, tk, tk + opts["dur"]))
            else:
                k = spring(t - tk, opts.get("freq", 3.0), opts.get("damp", 0.6))
            v = v + (target - v) * k
        return v


def pulse(t, t0, attack, hold, release, ease_in=ease_out_cubic, ease_out=ease_in_cubic):
    """0 -> 1 -> 0 envelope (visibility helper)."""
    if t <= t0:
        return 0.0
    if t < t0 + attack:
        return ease_in((t - t0) / attack)
    if t < t0 + attack + hold:
        return 1.0
    return 1.0 - ease_out(seg(t, t0 + attack + hold, t0 + attack + hold + release))
