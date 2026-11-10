"""Petrol-car swap: the electric cars in the source become gold petrol sedans.

Why this module exists
----------------------
The film is an engine-oil commercial, and the cars in shots 3, 8 and 9 are
battery-electric.  Their bodies are replaced with a gold-yellow BMW M5-style
petrol sedan (quad chrome tips, boot lip, fuel flap, red LED tail-lights).

How the swap is built
---------------------
There is no second take of this footage, so the replacement is a *patch*: one
still per key frame is re-drawn by an image edit (same frame in, same frame out,
only the car's body changed), the difference against that frame gives the car's
mask, and the patch is then carried to the neighbouring frames with a chained
optical-flow field measured on the *original* footage.  Because the flow is
measured on the real car, the patch follows the car's own motion, the camera's
move and the smoke around it; the paste itself is feathered and its exposure is
matched to the destination before compositing, so no seam survives the grade.

Shot 8 (the launch at the start line) is handled differently: the near car
leaves the frame there and its pose changes by more than a 2-D patch can carry,
so that beat is rebuilt from a single re-drawn plate that is pushed in slowly.
Everything else about the film - colour, timing, cuts - is untouched.
"""
import math
import os

import cv2
import numpy as np

SW, SH = 720, 1280                     # source resolution the module works at
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets", "cars")

# (shot, first frame, last frame, key frame, plate)
PLATES = [
    (3, 93, 116, 110, "shot3_a.jpg"),
    (3, 117, 136, 130, "shot3_b.jpg"),
    (9, 292, 360, 340, "shot9.jpg"),
]
HERO = (250, 283, "shot8_hero.jpg")


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _feather(mask, r=2.2):
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), r)
    return np.clip((m - 0.5) * 2.5 + 0.5, 0.0, 1.0)


def _exposure_match(patch, dst, alpha):
    """match the patch's exposure to the frame, measured *around* the car.

    The patch is a re-painted version of the same frame, so its paint must keep
    its own colour - only the drift of the optical-flow warp is corrected, and
    that drift is measured on the background ring just outside the body, which
    the re-paint left untouched.
    """
    m = (alpha > 0.02).astype(np.uint8)
    if m.sum() < 2000:
        return patch
    ring = cv2.subtract(cv2.dilate(m, np.ones((41, 41), np.uint8)),
                        cv2.dilate(m, np.ones((13, 13), np.uint8)))
    sel = ring > 0
    if sel.sum() < 2000:
        return patch
    w = np.array([0.114, 0.587, 0.299], np.float32)
    lum_p = (patch[sel] @ w) + 1e-3
    lum_d = (dst[sel] @ w) + 1e-3
    k = float(np.clip(np.median(lum_d) / np.median(lum_p), 0.80, 1.30))
    return np.clip(patch * k, 0, 255)


def _motion_blur(img, alpha, vel):
    """blur the patch along its own screen velocity (the plate is already blurred)"""
    length = float(np.hypot(*vel)) * 0.45
    if length < 1.5:
        return img
    k = min(int(round(length)), 41)
    kernel = np.zeros((k + 1, k + 1), np.float32)
    c = k / 2.0
    ang = math.atan2(vel[1], vel[0])
    dx, dy = math.cos(ang), math.sin(ang)
    cv2.line(kernel, (int(c - dx * k / 2), int(c - dy * k / 2)),
             (int(c + dx * k / 2), int(c + dy * k / 2)), 1.0, 1)
    kernel = kernel / max(kernel.sum(), 1e-6)
    blurred = cv2.filter2D(img, -1, kernel)
    mb = np.clip(cv2.filter2D(alpha, -1, kernel), 0.0, 1.0)
    return img * (1 - mb[..., None]) + blurred * mb[..., None]


def gold_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (10, 85, 55), (42, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def body_mask(plate, frame, thr=90, min_area=3000):
    """the car bodies a re-drawn plate actually changed.

    Reflections and lamp glows also differ, but they are *below* the body and
    therefore rejected: a component is only kept if it is tall enough to be a
    car, or if it touches a kept one (a car is often split in two by its own
    shadow line on the flank).
    """
    d = np.abs(frame.astype(np.int16) - plate.astype(np.int16)).sum(2).astype(np.float32)
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    comps = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < min_area:
            continue
        comps.append((k, x, y, w, h, a))
    if not comps:
        return np.zeros((SH, SW), np.uint8)
    tall = max(c[4] for c in comps)
    keep = {c[0] for c in comps if c[4] >= 0.45 * tall}
    # join the pieces that belong to the same body
    for _ in range(3):
        for k, x, y, w, h, a in comps:
            if k in keep:
                continue
            for k2, x2, y2, w2, h2, _a in comps:
                if k2 not in keep:
                    continue
                if (x - 10 < x2 + w2 and x2 - 10 < x + w and
                        y - 10 < y2 + h2 and y2 - 10 < y + h):
                    keep.add(k)
                    break
    out = np.zeros((SH, SW), np.uint8)
    for k in keep:
        cnts, _ = cv2.findContours(((lab == k).astype(np.uint8) * 255),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, 255, -1)
    # a solid body: close the gaps a shadow line cuts through a flank, then fill
    # every enclosed hole so no sliver of the original car can survive the paste
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    flood = out.copy()
    cv2.floodFill(flood, np.zeros((SH + 2, SW + 2), np.uint8), (0, 0), 255)
    core = cv2.bitwise_or(out, cv2.bitwise_not(flood))
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # the replacement is painted the same colour as the car it replaces, so the
    # difference leaves holes across the flank: the gold paint of the *source*
    # car, near the difference, is added to the mask
    ylim = int(max(c[2] + c[4] for c in comps if c[4] >= 0.45 * tall)) + 4
    above = np.zeros((SH, SW), np.uint8)
    above[:max(1, ylim), :] = 255
    gold = cv2.bitwise_and(gold_mask(frame), cv2.dilate(core, np.ones((61, 61), np.uint8)))
    gold = cv2.bitwise_and(gold, above)
    out = cv2.bitwise_or(core, gold)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    flood = out.copy()
    cv2.floodFill(flood, np.zeros((SH + 2, SW + 2), np.uint8), (0, 0), 255)
    out = cv2.bitwise_or(out, cv2.bitwise_not(flood))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # the car's own reflection on the wet floor stops at the wheels: anything
    # below that line is floor, not body, and must stay out of the snow mask
    core = cv2.bitwise_and(core, above)
    return out, core


def blacken(img, mask):
    """re-paint the gold bodies black, keeping their form and their highlights.

    The plates are re-drawn gold cars; the film needs a black one.  Mapping the
    body's own luminance through a steep curve keeps every panel line, every
    reflection and every specular streak, so the car still reads as a car - it is
    the same geometry, only the paint changed.  Red tail-lights are left alone.
    """
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    t = np.clip((0.299 * r + 0.587 * g + 0.114 * b) / 255.0, 0.0, 1.0)
    t = cv2.GaussianBlur(t, (0, 0), 4.0)           # smooth, so the paint is not blotchy
    v = 0.045 + 0.60 * np.clip((t - 0.26) / 0.60, 0.0, 1.0) ** 1.55
    paint = np.dstack([v * 0.96, v * 0.99, v * 1.07]) * 255.0
    red = (g < 0.35 * r) & (b < 0.35 * r) & (r > 80)   # tail-lights stay red
    body = np.where(red[..., None], f, paint)
    m = (mask > 0)[..., None]
    return np.where(m, body, f).astype(np.uint8)


def degold(img):
    """turn the gold alloy in the wheel close-up into dark gunmetal.

    The car is black now, so its wheels cannot stay gold.  Only the gold paint
    is touched: the tyre, the brake disc and the reflections keep their own
    pixels, and the rim keeps its highlights so the metal still turns.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    f = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (8, 70, 45), (45, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.0)[..., None]
    t = np.clip((0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]) / 255.0, 0, 1)
    t = cv2.GaussianBlur(t, (0, 0), 2.0)
    v = 0.055 + 0.72 * np.clip((t - 0.18) / 0.62, 0.0, 1.0) ** 1.5
    metal = np.dstack([v * 0.98, v * 1.00, v * 1.04]) * 255.0
    out = np.clip(f * (1 - a) + metal * a, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
#  one re-drawn plate carried across the frames it belongs to
# ---------------------------------------------------------------------------
class Patch:
    def __init__(self, raw, key, name):
        self.raw = raw
        self.key = key
        self.range = None
        bgr = np.asarray(raw[key])[:, :, ::-1].copy()
        ed = cv2.imread(os.path.join(ASSETS, name))
        if ed is None:
            raise FileNotFoundError(name)
        self.edit = cv2.resize(ed, (SW, SH), interpolation=cv2.INTER_AREA)
        self.mask, self.core = body_mask(self.edit, bgr)
        self.edit = blacken(self.edit, self.mask)
        self._maps = None
        self._last = (None, None)
        self._last_core = (None, None)

    # -- optical flow between two source frames (per-pixel, frame m -> frame n)
    def _step(self, m, n):
        a = cv2.cvtColor(np.asarray(self.raw[m])[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(np.asarray(self.raw[n])[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 5, 7, 1.5, 0)
        gx, gy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))
        return np.dstack([gx + fl[..., 0], gy + fl[..., 1]])

    def build(self, lo, hi):
        """chain the flow outward from the key frame; keep a quarter-res cache"""
        if self._maps is not None:
            return
        gx, gy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))
        small = (SW // 4, SH // 4)
        maps = {}
        ident = np.dstack([gx, gy]).copy()
        maps[self.key] = (cv2.resize(ident, small, interpolation=cv2.INTER_AREA) * 0.25,
                          (0.0, 0.0))
        for d in (-1, 1):
            M = ident.copy()
            n = self.key
            while lo <= n + d <= hi:
                m = n + d
                S = self._step(m, n)
                # translational part of this step, used for the motion blur
                vel = (float(np.median(S[..., 0] - gx)), float(np.median(S[..., 1] - gy)))
                M = cv2.remap(M, S[..., 0], S[..., 1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
                n = m
                maps[n] = (cv2.resize(M, small, interpolation=cv2.INTER_AREA) * 0.25, vel)
        self._maps = maps
        self.range = (lo, hi)

    def has(self, i):
        return self._maps is not None and i in self._maps

    def apply(self, img_bgr, i):
        sm, vel = self._maps[i]
        M = cv2.resize(sm, (SW, SH), interpolation=cv2.INTER_LINEAR) * 4.0
        Mx = np.ascontiguousarray(M[..., 0], dtype=np.float32)
        My = np.ascontiguousarray(M[..., 1], dtype=np.float32)
        col = cv2.remap(self.edit, Mx, My, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        msk = cv2.remap(self.mask, Mx, My, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self._last = (i, msk)
        self._last_core = (i, cv2.remap(self.core, Mx, My, cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        # the pasted body is a little larger than the one being replaced (two
        # different cars), so the alpha is grown: the extra ring lays the plate's
        # own background over the destination's background, which is invisible,
        # while no sliver of the old paint can survive at the silhouette
        msk = cv2.dilate(msk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        a = _feather(msk, 3.2)
        if a.max() < 0.02:
            return img_bgr
        col = _motion_blur(col.astype(np.float32), a, vel)
        dst = img_bgr.astype(np.float32)
        col = _exposure_match(col, dst, a)
        out = dst * (1 - a[..., None]) + col * a[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  the launch beat, rebuilt from a single plate
# ---------------------------------------------------------------------------
class Hero:
    def __init__(self, f0, f1, name, raw=None, key=258):
        self.f0, self.f1 = f0, f1
        im = cv2.imread(os.path.join(ASSETS, name))
        if im is None:
            raise FileNotFoundError(name)
        plate = cv2.resize(im, (SW, SH), interpolation=cv2.INTER_AREA)
        self.mask = None
        self.core = None
        if raw is not None:
            self.mask, self.core = body_mask(plate, np.asarray(raw[key])[:, :, ::-1].copy())
            plate = blacken(plate, self.mask)
        self.base = plate
        self._last = (None, None)

    def frame(self, i):
        t = (i - self.f0) / float(max(1, self.f1 - self.f0 - 1))
        u = t * t * (3.0 - 2.0 * t)                     # slow settle, no snap at the cut
        s = 1.0 + 0.095 * u
        cx, cy = SW * 0.5 + 12.0 * u, SH * 0.5 - 8.0 * u
        M = np.float32([[s, 0, cx - s * SW * 0.5], [0, s, cy - s * SH * 0.5]])
        out = cv2.warpAffine(self.base, M, (SW, SH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dx = 0.7 * math.sin(0.85 * (i - self.f0))
        dy = 0.5 * math.cos(0.63 * (i - self.f0))
        out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (SW, SH),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if self.core is not None:
            mk = cv2.warpAffine(self.core, M, (SW, SH), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mk = cv2.warpAffine(mk, np.float32([[1, 0, dx], [0, 1, dy]]), (SW, SH),
                                flags=cv2.INTER_NEAREST, borderValue=0)
            self._last = (i, mk)
        return out


# ---------------------------------------------------------------------------
#  public entry point (used by scene.Film.plate)
# ---------------------------------------------------------------------------
class Cars:
    def __init__(self, raw):
        self.raw = raw
        self._patches = {}
        self._hero = None
        self._cur = None

    def active(self, i):
        if HERO[0] <= i <= HERO[1]:
            return True
        return any(lo <= i <= hi for _s, lo, hi, _k, _n in PLATES)

    def _spec_for(self, i):
        for shot, lo, hi, key, name in PLATES:
            if lo <= i <= hi:
                return shot, lo, hi, key, name
        return None

    def _patch(self, spec):
        _shot, lo, hi, key, name = spec
        p = self._patches.get(key)
        if p is None:
            # the neighbouring plate of the same shot shares the chain work
            p = Patch(self.raw, key, name)
            p.build(lo, hi)
            self._patches[key] = p
        return p

    def cover(self, i):
        """silhouette of the cars in frame i (uint8 0/255), or None"""
        if HERO[0] <= i <= HERO[1]:
            if self._hero is None:
                self._hero = Hero(HERO[0], HERO[1], HERO[2], raw=self.raw, key=258)
            idx, mk = self._hero._last
            return mk if idx == i else None
        spec = self._spec_for(i)
        if spec is None:
            return None
        p = self._patches.get(spec[3])
        if p is None or not p.has(i):
            return None
        idx, mk = p._last_core
        return mk if idx == i else None

    def apply(self, img, i):
        """img: RGB uint8 (SH, SW, 3) -> RGB uint8, cars swapped"""
        if HERO[0] <= i <= HERO[1]:
            if self._hero is None:
                self._hero = Hero(HERO[0], HERO[1], HERO[2], raw=self.raw, key=258)
            return cv2.cvtColor(self._hero.frame(i), cv2.COLOR_BGR2RGB)
        spec = self._spec_for(i)
        if spec is None:
            return img
        p = self._patch(spec)
        if not p.has(i):
            return img
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = p.apply(bgr, i)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
