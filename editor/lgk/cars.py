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


def _colour_match(patch, dst, alpha):
    """match the patch's exposure and contrast to the plate it lands on"""
    sel = alpha > 0.5
    if sel.sum() < 200:
        return patch
    a, b = patch[sel], dst[sel]
    am, bm = a.mean(0), b.mean(0)
    asd, bsd = a.std(0) + 1e-3, b.std(0) + 1e-3
    k = np.clip(bsd / asd, 0.75, 1.33)
    return np.clip((patch - am) * k + bm, 0, 255)


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
    return out


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
        self.mask = body_mask(self.edit, bgr)
        self._maps = None

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
        a = _feather(msk, 2.2)
        if a.max() < 0.02:
            return img_bgr
        col = _motion_blur(col.astype(np.float32), a, vel)
        dst = img_bgr.astype(np.float32)
        col = _colour_match(col, dst, a)
        out = dst * (1 - a[..., None]) + col * a[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  the launch beat, rebuilt from a single plate
# ---------------------------------------------------------------------------
class Hero:
    def __init__(self, f0, f1, name):
        self.f0, self.f1 = f0, f1
        im = cv2.imread(os.path.join(ASSETS, name))
        if im is None:
            raise FileNotFoundError(name)
        self.base = cv2.resize(im, (SW, SH), interpolation=cv2.INTER_AREA)

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

    def apply(self, img, i):
        """img: RGB uint8 (SH, SW, 3) -> RGB uint8, cars swapped"""
        if HERO[0] <= i <= HERO[1]:
            if self._hero is None:
                self._hero = Hero(*HERO)
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
