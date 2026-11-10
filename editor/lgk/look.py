"""Camera moves (with real motion blur), colour grade and finishing passes."""
import math

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ------------------------------------------------------------------- camera
def _affine(sw, sh, ow, oh, zoom, rot, tx, ty):
    """Matrix mapping source pixel indices -> output pixel indices (centre pivot)."""
    s = (ow / sw) * zoom
    c, sn = math.cos(rot), math.sin(rot)
    a, b = s * c, -s * sn
    d, e = s * sn, s * c
    cx, cy = sw * 0.5, sh * 0.5
    ox, oy = ow * 0.5 + tx, oh * 0.5 + ty
    # out + 0.5 = A (src + 0.5 - c) + o
    return np.float32([[a, b, a * (0.5 - cx) + b * (0.5 - cy) + ox - 0.5],
                       [d, e, d * (0.5 - cx) + e * (0.5 - cy) + oy - 0.5]])


def warp(src, out_size, zoom=1.0, rot=0.0, tx=0.0, ty=0.0, ca=0.0, interp=cv2.INTER_LANCZOS4):
    """Resample a source frame through a virtual camera. `ca` splits R/B radially."""
    ow, oh = out_size
    sh, sw = src.shape[:2]
    if abs(ca) < 1e-5:
        M = _affine(sw, sh, ow, oh, zoom, rot, tx, ty)
        return cv2.warpAffine(src, M, (ow, oh), flags=interp, borderMode=cv2.BORDER_REFLECT_101)
    out = np.empty((oh, ow, 3), src.dtype)
    for c, k in enumerate((1.0 + ca, 1.0, 1.0 - ca)):
        M = _affine(sw, sh, ow, oh, zoom * k, rot, tx, ty)
        out[..., c] = cv2.warpAffine(np.ascontiguousarray(src[..., c]), M, (ow, oh), flags=interp,
                                     borderMode=cv2.BORDER_REFLECT_101)
    return out


def camera(src, out_size, cam_fn, shutter=0.5):
    """Render one frame through an animated camera.

    cam_fn(u) -> dict(zoom, rot, tx, ty, ca) for u in [0, 1) across the frame
    interval.  When the camera moves during the open shutter the frame is
    integrated over many sub-samples, which yields true zoom / spin blur.
    """
    p0 = cam_fn(0.0)
    p1 = cam_fn(shutter)
    ow, oh = out_size
    R = math.hypot(ow, oh) * 0.5
    travel = (abs(p1["zoom"] - p0["zoom"]) * R + abs(p1["rot"] - p0["rot"]) * R
              + math.hypot(p1["tx"] - p0["tx"], p1["ty"] - p0["ty"]))
    if travel < 0.75:
        return warp(src, out_size, **p0)
    n = int(min(64, max(6, math.ceil(travel / 5.0))))
    acc = np.zeros((oh, ow, 3), np.float32)
    for j in range(n):
        u = shutter * (j + 0.5) / n
        acc += warp(src, out_size, interp=cv2.INTER_LINEAR, **cam_fn(u))
    acc /= n
    return acc


def unsharp(img, sigma=1.1, amount=0.45):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return img + (img - blur) * amount


# -------------------------------------------------------------------- grade
def grade(x, contrast=0.16, sat=1.06, vibrance=0.18,
          shadow_tint=(-0.010, 0.000, 0.014), high_tint=(0.016, 0.006, -0.018),
          gold=0.10, exposure=1.0):
    """Cinematic finishing grade: filmic S-curve, cool shadows / warm highlights,
    vibrance and a selective push on the gold/amber range."""
    y = x * exposure if exposure != 1.0 else x
    y = np.clip(y, 0.0, 1.0)
    s = y * y * (3.0 - 2.0 * y)
    y = y + (s - y) * contrast
    l = y @ LUMA
    sh = (1.0 - l) ** 2
    hi = l * l
    y = y + sh[..., None] * np.asarray(shadow_tint, np.float32) + hi[..., None] * np.asarray(high_tint, np.float32)
    l = (y @ LUMA)[..., None]
    mx = y.max(axis=2, keepdims=True)
    mn = y.min(axis=2, keepdims=True)
    satn = (mx - mn) / (mx + 1e-4)
    k = sat * (1.0 + vibrance * (1.0 - satn))
    if gold > 0:
        # amber / gold: R > G > B with a decent spread
        r, g, b = y[..., 0:1], y[..., 1:2], y[..., 2:3]
        w = np.clip((r - b) * 2.2, 0, 1) * np.clip((g - b) * 3.0, 0, 1) * np.clip((r - g) * 6.0 + 0.4, 0, 1)
        k = k * (1.0 + gold * w)
    y = l + (y - l) * k
    return np.clip(y, 0.0, 1.0)


def bloom(x, thresh=0.74, amount=0.20, tint=(1.0, 0.90, 0.72)):
    """Soft highlight halation (computed at quarter resolution)."""
    H, W = x.shape[:2]
    small = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    l = small.max(axis=2)
    m = np.clip((l - thresh) / (1.0 - thresh), 0.0, 1.0)
    hl = small * (m * m)[..., None]
    b = 0.6 * cv2.GaussianBlur(hl, (0, 0), 5.0) + 0.4 * cv2.GaussianBlur(hl, (0, 0), 14.0)
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * np.asarray(tint, np.float32)
    return 1.0 - (1.0 - x) * (1.0 - np.clip(b * amount, 0.0, 1.0))


_vig_cache = {}


def vignette(x, amount=0.20):
    H, W = x.shape[:2]
    key = (H, W, amount)
    if key not in _vig_cache:
        ys = (np.arange(H, dtype=np.float32) + 0.5) / H * 2 - 1
        xs = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 - 1
        X, Y = np.meshgrid(xs, ys)
        r = np.sqrt((X * 0.92) ** 2 + (Y * 0.80) ** 2)
        t = np.clip((r - 0.55) / (1.22 - 0.55), 0.0, 1.0)
        _vig_cache[key] = (1.0 - amount * t * t * (3 - 2 * t))[..., None].astype(np.float32)
    return x * _vig_cache[key]


def flash(x, amount, color=(1.0, 0.94, 0.84)):
    """Exposure burst: highlights blow out first, blacks keep some density
    (reads as a light hit, not as a white fade)."""
    if amount <= 0.002:
        return x
    y = x * (1.0 + 2.4 * amount) + (0.16 * amount) * np.asarray(color, np.float32)
    knee = 0.78
    over = np.maximum(y - knee, 0.0)
    return np.minimum(y, knee) + (1.0 - knee) * np.tanh(over / (1.0 - knee))


class Grain:
    """Fine monochrome film grain, strongest in the mid-tones."""

    def __init__(self, size, plates=10, seed=7):
        W, H = size
        rng = np.random.default_rng(seed)
        self.plates = []
        for _ in range(plates):
            g = rng.standard_normal((H, W)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.62)
            g /= g.std()
            self.plates.append(g)
        self.rng = np.random.default_rng(seed + 1)

    def apply(self, x, i, amount=0.014):
        g = self.plates[i % len(self.plates)]
        k = (i * 2654435761) & 3
        if k & 1:
            g = g[:, ::-1]
        if k & 2:
            g = g[::-1]
        l = x @ LUMA
        w = 0.35 + 2.6 * l * (1.0 - l)
        return x + (g * w * amount)[..., None]


def to_u8(x, i=0):
    """Float -> 8 bit with a little ordered noise to avoid banding."""
    H, W = x.shape[:2]
    rng = np.random.default_rng(1234 + i)
    d = (rng.random((H, W, 1), dtype=np.float32) - 0.5) / 255.0
    return (np.clip(x + d, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
