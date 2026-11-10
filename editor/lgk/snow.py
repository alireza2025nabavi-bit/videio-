"""Falling snow: a depth-layered particle field, composited in-camera.

The field is generated once from a fixed seed and every flake is positioned from
absolute time, so the snowfall is continuous across cuts instead of restarting.
Flakes are grouped in depth layers: the further away, the smaller, slower and
dimmer they are; the closest ones are big, soft, out-of-focus bokeh that pass
the lens quickly and carry the shot's speed.

Compositing is a *screen* blend of neutral white, so the plate keeps its own
colour - the snow only adds light where a flake is in front of the lens. Each
flake's brightness follows the local exposure of the plate (flakes drifting in
front of a lamp bloom, flakes against a dark garage stay grey), which is what
makes the layer feel like it is inside the scene rather than pasted on top.
"""
import math

import cv2
import numpy as np


class Snow:
    def __init__(self, w, h, seed=1301, count=900, near=44):
        self.w, self.h = w, h
        rng = np.random.default_rng(seed)
        # ---------------------------------------------------------- far/mid field
        n = count
        d = rng.random(n) ** 0.75                     # depth 0 (far) .. 1 (near)
        self.x0 = rng.random(n) * (w + 400.0) - 200.0
        self.y0 = rng.random(n) * (h + 400.0) - 200.0
        self.r = 1.25 + 5.4 * d ** 1.7
        self.v = 52.0 + 168.0 * d ** 1.5              # px / s downwards
        self.wind = -(22.0 + 26.0 * d)                # px / s to the left
        self.sway = 3.0 + 22.0 * d
        self.freq = 0.18 + 0.55 * rng.random(n)       # Hz
        self.phase = rng.random(n) * 2.0 * math.pi
        self.alpha = 0.42 + 0.55 * d
        self.key = rng.random(n)                      # density culling order
        self.tw = 0.6 + 1.7 * rng.random(n)           # twinkle rate, Hz
        # ---------------------------------------------------------- near bokeh
        self.nx0 = rng.random(near) * (w + 500.0) - 250.0
        self.ny0 = rng.random(near) * (h + 500.0) - 250.0
        self.nr = 8.0 + 15.0 * rng.random(near)
        self.nv = 300.0 + 190.0 * rng.random(near)
        self.nwind = -(60.0 + 40.0 * rng.random(near))
        self.nalpha = 0.12 + 0.16 * rng.random(near)
        self.nkey = rng.random(near)
        self._kern = {}
        self._lum = None

    # ------------------------------------------------------------------ helpers
    def _kernel(self, sigma):
        q = round(sigma * 4.0) / 4.0
        k = self._kern.get(q)
        if k is None:
            rad = max(1, int(math.ceil(q * 2.6)))
            ax = np.arange(-rad, rad + 1, dtype=np.float32)
            g = np.exp(-0.5 * (ax / q) ** 2)
            k = np.outer(g, g)
            k *= (1.0 / k.max())
            self._kern[q] = k.astype(np.float32)
        return k

    def _splat(self, acc, x, y, k, inten):
        """Add one kernel at a sub-pixel position (bilinear weighted)."""
        kh, kw = k.shape
        h, w = acc.shape
        fx, fy = x - math.floor(x), y - math.floor(y)
        for ox, oy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                            (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if wgt <= 0.004:
                continue
            x0 = int(math.floor(x)) + ox - kw // 2
            y0 = int(math.floor(y)) + oy - kh // 2
            x1, y1 = min(x0 + kw, w), min(y0 + kh, h)
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                continue
            sx0, sy0 = max(0, -x0), max(0, -y0)
            x0, y0 = max(0, x0), max(0, y0)
            acc[y0:y1, x0:x1] += k[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] * (inten * wgt)

    # --------------------------------------------------------------------- draw
    def layer(self, t, amount=1.0):
        """Snow intensity map (0..1) for time t, density scaled by `amount`."""
        acc = np.zeros((self.h, self.w), np.float32)
        span = float(self.h + 500.0)
        # far / mid
        for i in range(len(self.r)):
            a = self.alpha[i] * np.clip((amount * 1.55 - self.key[i]) * 2.6, 0.0, 1.0)
            if a <= 0.004:
                continue
            sw = math.sin(2.0 * math.pi * self.freq[i] * t + self.phase[i])
            x = self.x0[i] + self.wind[i] * t + self.sway[i] * sw
            x = (x + 250.0) % (self.w + 500.0) - 250.0
            y = (self.y0[i] + self.v[i] * t + 0.30 * self.sway[i] * math.cos(
                2.0 * math.pi * self.freq[i] * t + self.phase[i])) % span - 250.0
            if y < -20.0 or y > self.h + 20.0:
                continue
            tw = 0.82 + 0.18 * math.sin(2.0 * math.pi * self.tw[i] * t + self.phase[i] * 2.0)
            self._splat(acc, x, y, self._kernel(self.r[i] * 0.55), a * tw)
        # near bokeh: wider drift, softer, dimmer
        for i in range(len(self.nr)):
            a = self.nalpha[i] * np.clip((amount * 1.55 - self.nkey[i]) * 2.6, 0.0, 1.0)
            if a <= 0.004:
                continue
            x = self.nx0[i] + self.nwind[i] * t
            x = (x + 300.0) % (self.w + 600.0) - 300.0
            y = (self.ny0[i] + self.nv[i] * t) % (self.h + 600.0) - 300.0
            if y < -60.0 or y > self.h + 60.0:
                continue
            self._splat(acc, x, y, self._kernel(self.nr[i] * 0.60), a)
        return acc

    def draw(self, img, t, amount=1.0, gain=1.0):
        """Composite snow over an RGB float image (0..1). `img` is modified in place."""
        if amount <= 0.0:
            return img
        acc = self.layer(t, amount)
        if gain != 1.0:
            acc *= gain
        # exposure coupling: flakes pick up light from the plate behind them
        lum = cv2.resize(img.mean(axis=2), (self.w // 8, self.h // 8), interpolation=cv2.INTER_AREA)
        lum = cv2.GaussianBlur(lum, (0, 0), 3.0)
        lum = cv2.resize(lum, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mod = 0.72 + 0.34 * np.clip(lum, 0.0, 1.2)
        s = np.clip(acc * mod, 0.0, 1.0)[..., None]
        img *= (1.0 - s)
        img += s
        return img
