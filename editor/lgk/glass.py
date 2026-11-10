"""Liquid-glass material renderer.

Shapes are described by signed distance fields (negative inside, in pixels).
From the SDF we derive, per pixel:

  * coverage (anti-aliased edge)
  * the outward surface normal -> a lens-like bevel that *refracts* the
    footage behind it (with a little chromatic dispersion)
  * a specular rim that reacts to a light direction
  * a soft contact shadow

Because everything is a distance field, shapes can be blended with a smooth
minimum which makes them merge and split like real droplets.
"""
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# --------------------------------------------------------------------- SDFs
def sd_rrect(X, Y, cx, cy, w, h, r, p=2.0):
    """Rounded rectangle. p > 2 gives slightly 'squircle' (continuous) corners."""
    w = max(w, 0.0)
    h = max(h, 0.0)
    r = max(0.0, min(r, w * 0.5, h * 0.5))
    qx = np.abs(X - cx) - (w * 0.5 - r)
    qy = np.abs(Y - cy) - (h * 0.5 - r)
    ax = np.maximum(qx, 0.0)
    ay = np.maximum(qy, 0.0)
    if p == 2.0:
        outside = np.sqrt(ax * ax + ay * ay)
    else:
        outside = (ax ** p + ay ** p) ** (1.0 / p)
    return outside + np.minimum(np.maximum(qx, qy), 0.0) - r


def sd_circle(X, Y, cx, cy, r):
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r


def smin(a, b, k):
    """Polynomial smooth minimum: fuses two shapes with a liquid neck of size ~k."""
    if k <= 1e-4:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


# -------------------------------------------------------------------- style
@dataclass
class Glass:
    bevel: float = 26.0          # width of the curved lens edge (px)
    refr: float = 30.0           # how far the edge bends the background (px)
    curve: float = 1.7           # fall-off of the bending through the bevel
    disp: float = 0.11           # chromatic dispersion (fraction of refr)
    blur: float = 5.0            # frost amount of the interior (gaussian sigma, px)
    edge_blur: float = 1.1       # the lens edge stays (almost) clear so refraction reads crisp
    sat: float = 1.25            # vibrancy of what is seen through the glass
    dim: float = 0.18            # darkening of what is seen through the glass
    adapt: float = 0.0           # extra darkening on bright backgrounds (keeps text readable)
    tint: tuple = (1.0, 1.0, 1.0)
    tint_a: float = 0.03
    glow: float = 0.10           # fresnel-like brightening inside the bevel
    glow_col: tuple = (1.0, 1.0, 1.0)
    rim: float = 0.9             # specular rim strength
    rim_w: float = 1.45          # rim line half-width (px)
    rim_soft: float = 0.30       # soft halo around the bright parts of the rim
    rim_soft_w: float = 5.0
    rim_base: float = 0.20       # rim visibility where it does not face the light
    rim_pow: float = 2.4
    rim_back: float = 0.75       # strength of the opposite (bottom-right) rim
    rim_col: tuple = (1.0, 0.985, 0.95)
    edge_shade: float = 0.10     # soft inner shading of the bevel away from light
    inset: float = 0.16          # soft dark inner edge (gives the slab its thickness)
    inset_w: float = 9.0
    sheen: float = 0.06          # soft white wash on the upper part
    shadow_a: float = 0.30
    shadow_sigma: float = 22.0
    shadow_dy: float = 14.0
    outline: float = 0.18        # thin dark contact line just outside the edge
    light: tuple = (-0.6428, -0.7660)   # unit vector pointing toward the light (top-left)
    mag: float = 1.0             # lens magnification of the interior

    def but(self, **kw):
        return replace(self, **kw)


def mix_style(a: Glass, b: Glass, t: float) -> Glass:
    """Linear blend of two glass styles (numeric fields only)."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    out = {}
    for k, va in a.__dict__.items():
        vb = b.__dict__[k]
        if isinstance(va, tuple):
            out[k] = tuple(x + (y - x) * t for x, y in zip(va, vb))
        else:
            out[k] = va + (vb - va) * t
    return Glass(**out)


# ------------------------------------------------------------------ helpers
def padded_crop(img, x0, y0, x1, y1):
    """Crop that may extend beyond the image (edge pixels are replicated)."""
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    c = img[cy0:cy1, cx0:cx1]
    if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
        c = cv2.copyMakeBorder(c, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1, cv2.BORDER_REPLICATE)
    return c


def frost(crop, sigma):
    """Gaussian blur in (approximately) linear light so highlights stay bright."""
    if sigma <= 0.3:
        return crop
    lin = crop * crop
    lin = cv2.GaussianBlur(lin, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(np.maximum(lin, 0.0))


# ----------------------------------------------------------------- renderer
def render_glass(dst, src, sdf_fn, bbox, st: Glass, opacity=1.0, box=None, sweep=None, center=None):
    """Composite a liquid-glass shape onto `dst`.

    dst      float32 HxWx3 canvas (modified in place)
    src      float32 HxWx3 image that is seen through the glass
    sdf_fn   callable (X, Y) -> signed distance in px
    bbox     (x0, y0, x1, y1) rough bounds of the shape
    box      (top, height) of the shape, used for the vertical sheen
    sweep    dict(pos, angle, width, gain): travelling specular highlight
    center   (cx, cy) pivot for interior magnification
    """
    if opacity <= 0.004:
        return
    H, W = dst.shape[:2]
    pad = int(st.shadow_sigma * 2.6 + abs(st.shadow_dy) + 6)
    x0 = max(0, int(math.floor(bbox[0])) - pad)
    y0 = max(0, int(math.floor(bbox[1])) - pad)
    x1 = min(W, int(math.ceil(bbox[2])) + pad)
    y1 = min(H, int(math.ceil(bbox[3])) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys)
    d = sdf_fn(X, Y).astype(np.float32)
    alpha = np.clip(0.5 - d, 0.0, 1.0)
    if float(alpha.max()) <= 0.0:
        return

    # ---- geometry
    gy, gx = np.gradient(d)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx = gx / gl
    ny = gy / gl
    s = np.maximum(-d, 0.0)                       # depth inside the shape
    inradius = float(s.max())
    bevel = max(2.0, min(st.bevel, inradius * 0.94))
    u = np.clip(s / bevel, 0.0, 1.0)
    edge = 1.0 - u                                # 1 at the edge -> 0 on the flat top
    prof = edge ** st.curve
    refr_amt = st.refr * min(1.0, bevel / max(st.bevel, 1e-3))

    # ---- what we see through the glass (refraction + frost + dispersion)
    m = int(refr_amt * (1.0 + st.disp) + 3.0 * st.blur + 6)
    crop = padded_crop(src, x0 - m, y0 - m, x1 + m, y1 + m)
    if crop is None:
        return
    bx = X - (x0 - m) - 0.5
    by = Y - (y0 - m) - 0.5
    if center is not None and abs(st.mag - 1.0) > 1e-3:
        k = 1.0 / st.mag - 1.0
        bx = bx + (X - center[0]) * k
        by = by + (Y - center[1]) * k

    def lookup(img):
        if st.disp > 1e-3 and refr_amt > 0.5:
            chans = []
            for c, kk in enumerate((1.0 - st.disp, 1.0, 1.0 + st.disp)):
                D = prof * (refr_amt * kk)
                chans.append(cv2.remap(np.ascontiguousarray(img[..., c]), bx - nx * D, by - ny * D,
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            return np.stack(chans, axis=-1)
        D = prof * refr_amt
        return cv2.remap(img, bx - nx * D, by - ny * D, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    g = lookup(frost(crop, st.blur))
    if st.blur - st.edge_blur > 0.5:
        clear = lookup(frost(crop, st.edge_blur))
        k = (edge * edge * (3.0 - 2.0 * edge))[..., None]      # 1 at the rim -> 0 inside
        g = g + (clear - g) * k

    # ---- material: vibrancy, smoke, tint
    if abs(st.sat - 1.0) > 1e-3:
        l = (g @ LUMA)[..., None]
        g = l + (g - l) * st.sat
    dim = st.dim
    if st.adapt > 0:
        inside = alpha > 0.5
        if inside.any():
            lum = float((g[inside] @ LUMA).mean())
            tt = min(1.0, max(0.0, (lum - 0.30) / 0.45))
            dim = dim + st.adapt * tt * tt * (3.0 - 2.0 * tt)
    g = g * (1.0 - dim) + np.asarray(st.tint, np.float32) * st.tint_a

    # ---- lighting
    Lx, Ly = st.light
    ndl = nx * Lx + ny * Ly                       # +1 where the edge faces the light
    facing = np.maximum(ndl, 0.0)
    away = np.maximum(-ndl, 0.0)
    if st.edge_shade > 0:
        # subtle volume: the bevel gets darker where it turns sideways to the light
        side = 1.0 - np.abs(ndl)
        g = g * (1.0 - st.edge_shade * (edge ** 1.3) * side)[..., None]
    if st.inset > 0:
        g = g * (1.0 - st.inset * np.exp(-s / st.inset_w))[..., None]
    if st.glow > 0:
        gl_amt = st.glow * (edge ** 2.4) * (0.30 + 0.70 * (facing + 0.8 * away))
        g = g + (1.0 - g) * gl_amt[..., None] * np.asarray(st.glow_col, np.float32)
    if st.sheen > 0 and box is not None:
        tt = np.clip((Y - box[0]) / max(box[1] * 0.62, 1.0), 0.0, 1.0)
        g = g + (1.0 - g) * (st.sheen * (1.0 - tt) ** 2)[..., None]
    if sweep is not None and sweep.get("gain", 0) > 0.003:
        ang = sweep.get("angle", math.radians(22))
        ca, sa = math.cos(ang), math.sin(ang)
        lo = bbox[0] * ca + bbox[1] * sa
        hi = bbox[2] * ca + bbox[3] * sa
        wd = sweep.get("width", 90.0)
        pos = lo - 2 * wd + (hi - lo + 4 * wd) * sweep["pos"]
        band = np.exp(-(((X * ca + Y * sa) - pos) / wd) ** 2)
        band = band * (0.45 + 0.9 * edge ** 2)
        g = g + (1.0 - g) * np.clip(band * sweep["gain"], 0, 1)[..., None]
    if st.rim > 0:
        line = np.exp(-(((s - st.rim_w * 0.9) / st.rim_w) ** 2))
        spec = facing ** st.rim_pow + st.rim_back * away ** st.rim_pow
        ri = st.rim * (st.rim_base + (1.0 - st.rim_base) * spec)
        amt = line * ri
        if st.rim_soft > 0:
            amt = amt + st.rim * st.rim_soft * spec * np.exp(-s / st.rim_soft_w)
        g = g + (1.0 - g) * np.clip(amt, 0.0, 1.0)[..., None] * np.asarray(st.rim_col, np.float32)

    # ---- shadow + contact line on the outside, then composite
    roi = dst[y0:y1, x0:x1]
    outside = 1.0 - alpha
    if st.shadow_a > 0:
        k = int(round(st.shadow_dy))
        if k > 0:
            dsh = np.empty_like(d)
            dsh[k:] = d[:-k]
            dsh[:k] = d[:1]
        else:
            dsh = d
        sh = st.shadow_a * 0.5 * (1.0 - np.tanh(dsh / (0.85 * st.shadow_sigma)))
        roi *= (1.0 - sh * outside * opacity)[..., None]
    if st.outline > 0:
        ol = st.outline * np.exp(-(((d - 0.7) / 1.2) ** 2)) * outside * opacity
        roi *= (1.0 - ol)[..., None]
    a = (alpha * opacity)[..., None]
    roi *= (1.0 - a)
    roi += g * a
