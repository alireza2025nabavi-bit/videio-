"""The edit itself: timing map of the film + every liquid-glass element.

All times are in seconds, all coordinates in output pixels (1080 x 1920).
The cut list and the beat grid were measured from the source file:
  * 9 shots, hard cuts at the frame numbers in CUTS
  * music at 128 BPM, first down-beat at 0.119 s  ->  beat(k)
"""
import math

import cv2
import numpy as np

from . import cars as CR
from . import fix as FX
from . import glass as G
from . import look as L
from . import snow as SN
from . import typo as T
from .anim import (Track, clamp, ease_in_cubic, ease_in_expo, ease_in_out_cubic, ease_in_quad,
                   ease_out_cubic, ease_out_expo, ease_out_quad, lerp, seg, smooth, spring)

W, H, FPS, NFRAMES = 1080, 1920, 24, 361
CUTS = [0, 51, 93, 137, 176, 190, 220, 250, 284, 361]
BPM, BEAT0 = 128.0, 0.119
SHUTTER = 0.6            # camera moves
UI_SHUTTER = 0.45        # glass / UI motion
UI_MAX_SAMPLES = 30
GRAIN = 0.009
# snowfall density per shot (0 = clear, 1 = heavy); the weather runs through the
# whole film, heaviest out on the track and around the frozen garage
SNOW = [0.55, 0.30, 1.00, 0.85, 0.70, 0.20, 0.35, 0.95, 0.70]


def beat(k):
    return BEAT0 + k * 60.0 / BPM


def fr(n):
    """frame number -> seconds"""
    return n / FPS


def shot_of(i):
    for k in range(len(CUTS) - 1):
        if CUTS[k] <= i < CUTS[k + 1]:
            return k
    return len(CUTS) - 2


# =========================================================================
#  COPY  (everything on screen comes from the pack / the original end card)
# =========================================================================
BRAND = ("PAMIR", "LUBRICANTS")
INTRO = dict(kicker="PAMIR LUBRICANTS", headline="5W-30", sub="PETROL ENGINE OIL")
SPECS = [("SAE", "5W-30"), ("API", "SN"), ("SIZE", "4 LITER")]
FEATURES = [
    dict(icon="drop", kicker="PAMIR OIL  ·  01", label="HIGH QUALITY", t_in=fr(56), t_out=fr(85)),
    dict(icon="chevrons", kicker="PAMIR OIL  ·  02", label="PREMIUM PERFORMANCE", t_in=fr(97), t_out=fr(119.5)),
    dict(icon="shield", kicker="PAMIR OIL  ·  03", label="HIGH DURABILITY", t_in=fr(191.5), t_out=fr(224)),
]


# =========================================================================
#  CAMERA  (transitions + beat accents; timing of the cuts is untouched)
# =========================================================================
# zoom-through transitions: (cut frame, frames out, frames in, zoom out, zoom in, roll deg)
ZOOM_CUTS = [
    (51, 3, 4, 1.38, 0.70, 0.0),
    (176, 2, 3, 1.55, 0.66, 3.5),
    (190, 2, 3, 1.55, 0.66, -4.5),
    (220, 3, 3, 1.60, 0.64, 3.5),
]
# impact cuts: (cut frame, punch zoom, frames, chroma split, flash, shake px)
IMPACTS = [
    (93, 1.16, 7, 0.010, 0.50, 7.0),
    (250, 1.09, 6, 0.006, 0.00, 4.0),
    (284, 1.045, 9, 0.003, 0.14, 0.0),
]
# beat accents (small punch-in that decays): beat index -> strength
PULSES = {16: 0.028, 17: 0.030, 18: 0.030, 20: 0.030, 21: 0.030, 23: 0.026, 24: 0.026}
PULSE_TAU = 0.075


def cam_params(i, u):
    """Virtual camera for output frame i at sub-frame position u (0..1)."""
    f = i + u
    t = f / FPS
    zoom, rot, tx, ty, ca = 1.0, 0.0, 0.0, 0.0, 0.0
    for c, n_out, n_in, z_out, z_in, roll in ZOOM_CUTS:
        if c - n_out <= i < c:
            q = (f - (c - n_out)) / n_out
            e = ease_in_cubic(q)
            zoom *= 1.0 + (z_out - 1.0) * e
            rot += math.radians(roll) * e
            ca += 0.010 * q * q
        elif c <= i < c + n_in:
            q = (f - c) / n_in
            e = ease_out_expo(q)
            zoom *= z_in + (1.0 - z_in) * e
            rot -= math.radians(roll) * (1.0 - e)
            ca += 0.010 * (1.0 - q) ** 2
    for c, z0, n, split, _fl, shake in IMPACTS:
        if c <= i < c + n:
            q = (f - c) / n
            e = 1.0 - ease_out_cubic(q)
            zoom *= 1.0 + (z0 - 1.0) * e
            ca += split * e
            if shake:
                dec = math.exp(-q * 4.5)
                tx += shake * dec * math.sin(2 * math.pi * 11.0 * t + 1.3)
                ty += shake * dec * math.cos(2 * math.pi * 8.5 * t + 0.4)
    for k, amp in PULSES.items():
        tb = round(beat(k) * FPS) / FPS
        if tb <= t < tb + 0.45:
            zoom *= 1.0 + amp * math.exp(-(t - tb) / PULSE_TAU)
    return dict(zoom=zoom, rot=rot, tx=tx, ty=ty, ca=ca)


def flash_amount(i):
    a = 0.0
    for c, _z, _n, _s, fl, _sh in IMPACTS:
        if fl > 0 and c <= i < c + 6:
            a = max(a, fl * math.exp(-(i - c) * 0.95))
    # tiny exposure lift on the zoom-through cuts
    for c, n_out, n_in, *_ in ZOOM_CUTS:
        if i == c - 1 or i == c:
            a = max(a, 0.07)
    return a


# =========================================================================
#  LIQUID WAVE WIPE  (race -> product hero, lands on beat 12)
# =========================================================================
WIPE_F0, WIPE_F1 = 125.0, 137.0
WIPE_SHUTTER = 0.07          # short shutter: the liquid edge must stay readable
GOLD_LIQ = np.array([1.0, 0.74, 0.22], np.float32)


def _wipe_state(f):
    q = clamp((f - WIPE_F0) / (WIPE_F1 - WIPE_F0))
    e = q ** 2.0
    y0 = (H + 150.0) - (H + 150.0 + 260.0) * e
    return y0, f * 0.55


def _surface(xs, y0, ph):
    k = 2.0 * math.pi / W
    y = (y0 + 58.0 * np.sin(xs * k * 1.10 + ph) + 21.0 * np.sin(xs * k * 2.60 - 1.7 * ph + 1.0)
         + 8.0 * np.sin(xs * k * 5.20 + 2.3 * ph + 0.5))
    dy = (58.0 * k * 1.10 * np.cos(xs * k * 1.10 + ph) + 21.0 * k * 2.60 * np.cos(xs * k * 2.60 - 1.7 * ph + 1.0)
          + 8.0 * k * 5.20 * np.cos(xs * k * 5.20 + 2.3 * ph + 0.5))
    return y, dy


def _wipe_once(A, B, y0, ph, bevel=96.0, refr=150.0, curve=2.2, disp=0.10):
    xs = np.arange(W, dtype=np.float32) + 0.5
    ysurf, slope = _surface(xs, y0, ph)
    top = int(math.floor(float(ysurf.min()) - 110))
    bot = int(math.ceil(float(ysurf.max()) + bevel + 4))
    out = A.copy()
    if bot < H:
        out[max(bot, 0):] = B[max(bot, 0):]
    r0, r1 = max(top, 0), min(bot, H)
    if r1 <= r0:
        return out
    ys = np.arange(r0, r1, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys)
    norm = np.sqrt(1.0 + slope * slope)[None, :]
    d = (ysurf[None, :] - Y) / norm                      # >0 above the surface (outside)
    nx = (slope[None, :] / norm) * np.ones_like(Y)
    ny = (-1.0 / norm) * np.ones_like(Y)
    alpha = np.clip(0.5 - d, 0.0, 1.0)
    s = np.maximum(-d, 0.0)
    u = np.clip(s / bevel, 0.0, 1.0)
    edge = 1.0 - u
    prof = edge ** curve
    chans = []
    for c, kk in enumerate((1.0 - disp, 1.0, 1.0 + disp)):
        D = prof * refr * kk
        chans.append(cv2.remap(np.ascontiguousarray(B[..., c]), X - 0.5 - nx * D, Y - 0.5 - ny * D,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
    g = np.stack(chans, axis=-1)
    # liquid body: warm amber tint that is densest at the meniscus
    g = g * (1.0 - 0.10 * edge)[..., None]
    g = g + (1.0 - g) * (0.30 * edge ** 1.6)[..., None] * GOLD_LIQ
    g = g * (1.0 - 0.22 * np.exp(-s / 12.0))[..., None]
    # meniscus highlight
    line = np.exp(-((s - 2.0) / 2.0) ** 2) + 0.42 * np.exp(-s / 13.0)
    lite = 0.55 + 0.45 * np.clip(-nx * 0.64 - ny * 0.77, 0.0, 1.0)
    g = g + (1.0 - g) * np.clip(line * lite, 0, 1)[..., None] * np.float32([1.0, 0.97, 0.88])
    # shadow the liquid casts on the outgoing shot
    a_roi = out[r0:r1]
    sh = 0.50 * np.exp(-np.maximum(d, 0.0) / 34.0) * (1.0 - alpha)
    a_roi *= (1.0 - sh)[..., None]
    a3 = alpha[..., None]
    a_roi *= (1.0 - a3)
    a_roi += g * a3
    return out


def wave_wipe(A, B, i):
    """Outgoing shot A is flooded from below by B behind a refracting liquid edge."""
    y_a, _ = _wipe_state(i)
    y_b, _ = _wipe_state(i + WIPE_SHUTTER)
    n = int(min(24, max(1, math.ceil(abs(y_b - y_a) / 2.0))))
    acc = None
    for j in range(n):
        y0, ph = _wipe_state(i + WIPE_SHUTTER * (j + 0.5) / n)
        o = _wipe_once(A, B, y0, ph)
        acc = o if acc is None else acc + o
    return acc / n if n > 1 else acc


# =========================================================================
#  GLASS STYLES
# =========================================================================
ST_CARD = G.Glass(bevel=46, refr=92, curve=2.4, blur=3.5, dim=0.17, adapt=0.25, glow=0.17,
                  shadow_sigma=32, shadow_dy=22, shadow_a=0.36, sheen=0.07)
ST_ISLAND = G.Glass(bevel=29, refr=54, curve=2.4, blur=5.0, dim=0.24, adapt=0.40, glow=0.17,
                    shadow_sigma=20, shadow_dy=12, shadow_a=0.30)
ST_SPEC = ST_ISLAND.but(bevel=34, refr=60, blur=6.0, glow=0.30, sheen=0.14, adapt=0.30, sat=1.5)
ST_PILL = G.Glass(bevel=30, refr=58, curve=2.4, blur=10.0, dim=0.27, adapt=0.25, glow=0.17,
                  shadow_sigma=22, shadow_dy=14, shadow_a=0.32)
ST_PLATE = G.Glass(bevel=42, refr=58, curve=2.2, blur=0.0, edge_blur=0.0, dim=0.05, adapt=0.0, glow=0.16,
                   tint_a=0.02, sheen=0.05, shadow_sigma=34, shadow_dy=24, shadow_a=0.34, inset=0.12)


# =========================================================================
#  ISLAND  (one piece of liquid glass that lives at the top of the frame and
#           morphs: title card -> brand pill -> spec bar that splits in three
#           -> brand pill -> plaque behind the end lock-up)
# =========================================================================
ISL_Y = 285.0
T_APPEAR = fr(3)            # first lamp switches on / first sound
T_COLLAPSE = fr(45.5)       # just before the punch-in of shot 1
T_SPEC = fr(137) + 0.004    # the wipe has landed (beat 12)
T_SPLIT = beat(13) - 0.02
T_MERGE = fr(171.5)
T_SPEC_END = fr(173.5)
T_PLATE = fr(300)
T_SWEEP = beat(28)          # the big hit of the end card
PLATE = dict(cx=495.0, cy=548.0, w=676.0, h=508.0, r=66.0)

_SP_OPEN = dict(freq=2.7, damp=0.62)
_SP_CLOSE = dict(freq=3.0, damp=0.80)
_SP_SPEC = dict(freq=3.1, damp=0.64)
_SP_PLATE = dict(freq=2.3, damp=0.70)


class Island:
    def __init__(self):
        self.cx = Track(540.0, [(T_PLATE, PLATE["cx"], _SP_PLATE)])
        self.cy = Track(ISL_Y, [(T_APPEAR, 432.0, _SP_OPEN), (T_COLLAPSE, ISL_Y, _SP_CLOSE),
                                (T_SPEC, 292.0, _SP_SPEC), (T_SPEC_END, ISL_Y, _SP_CLOSE),
                                (T_PLATE, PLATE["cy"], _SP_PLATE)])
        self.w = Track(0.0, [(T_APPEAR, 880.0, _SP_OPEN), (T_COLLAPSE, 430.0, _SP_CLOSE),
                             (T_SPEC, 900.0, _SP_SPEC), (T_SPEC_END, 430.0, _SP_CLOSE),
                             (T_PLATE, PLATE["w"], _SP_PLATE)])
        self.h = Track(0.0, [(T_APPEAR + 0.03, 364.0, _SP_OPEN), (T_COLLAPSE, 72.0, _SP_CLOSE),
                             (T_SPEC, 112.0, _SP_SPEC), (T_SPEC_END, 72.0, _SP_CLOSE),
                             (T_PLATE + 0.04, PLATE["h"], _SP_PLATE)])
        e = dict(dur=0.30, ease=ease_out_cubic)
        self.r = Track(36.0, [(T_APPEAR, 72.0, e), (T_COLLAPSE, 36.0, e), (T_SPEC, 56.0, e),
                              (T_SPEC_END, 36.0, e), (T_PLATE, PLATE["r"], e)])
        # 0 = one bar, 1 = three separate pills
        self.split = Track(0.0, [(T_SPLIT, 1.0, dict(freq=3.0, damp=0.52)),
                                 (T_MERGE, 0.0, dict(dur=0.13, ease=ease_in_out_cubic))])
        # style morph weights
        m = dict(dur=0.28, ease=ease_out_cubic)
        self.k_card = Track(1.0, [(T_COLLAPSE, 0.0, m)])
        self.k_spec = Track(0.0, [(T_SPEC, 1.0, m), (T_SPEC_END, 0.0, m)])
        self.k_plate = Track(0.0, [(T_PLATE, 1.0, dict(dur=0.45, ease=ease_out_cubic))])
        self._build_sprites()

    # ----------------------------------------------------------- sprites
    def _build_sprites(self):
        sh = lambda sp, b=5.0, a=0.50: T.with_shadow(sp, b, 2.0, a)
        self.s_kicker = sh(T.text(INTRO["kicker"], 23, 700, False, 0.34, T.GOLD))
        self.s_kdrop = T.icon("drop", 30)
        self.s_head = T.with_shadow(T.text(INTRO["headline"], 158, 800, True, -0.012, T.WHITE), 12.0, 5.0, 0.42)
        self.s_sub = sh(T.text(INTRO["sub"], 27, 600, False, 0.42, T.WHITE, 0.94))
        self.s_line = T.bar(96, 3, T.GOLD_GRAD[1][1], 0.95)

        self.s_b1 = sh(T.text(BRAND[0], 24, 700, False, 0.17, T.WHITE))
        self.s_b2 = sh(T.text(BRAND[1], 19.5, 500, False, 0.26, T.WHITE, 0.82))
        self.s_bdrop = T.icon("drop", 30)
        self.s_bdiv = T.bar(2, 22, T.WHITE, 0.34)

        self.s_spec = []
        for lab, val in SPECS:
            self.s_spec.append((sh(T.text(lab, 15.5, 700, False, 0.34, T.GOLD), 4.0),
                                sh(T.text(val, 35, 800, False, 0.02, T.WHITE), 6.0)))

    # ------------------------------------------------------------- state
    def geo(self, t):
        return dict(cx=self.cx(t), cy=self.cy(t), w=self.w(t), h=self.h(t), r=self.r(t), split=self.split(t))

    def travel(self, t0, t1):
        if t1 <= T_APPEAR:
            return 0.0
        a, b = self.geo(t0), self.geo(t1)
        d = max(abs(a["cx"] - b["cx"]) + abs(a["w"] - b["w"]) * 0.5,
                abs(a["cy"] - b["cy"]) + abs(a["h"] - b["h"]) * 0.5)
        d = max(d, abs(a["split"] - b["split"]) * (24.0 + 2.0 * a["r"]) * 0.75)
        sa, sb = self._sweep(t0), self._sweep(t1)
        if sa[1] > 0 or sb[1] > 0:
            d = max(d, abs(sa[0] - sb[0]) * a["w"] * 0.12, 3.0)
        return d

    def style(self, t):
        st = G.mix_style(ST_ISLAND, ST_CARD, clamp(self.k_card(t)))
        st = G.mix_style(st, ST_SPEC, clamp(self.k_spec(t)))
        st = G.mix_style(st, ST_PLATE, clamp(self.k_plate(t)))
        return st

    def _sweep(self, t):
        # travelling highlight: as the title lands, on the spec pills, and on the final hit
        for t0, dur, gain in ((beat(1) - 0.02, 0.60, 0.34), (beat(14) - 0.05, 0.44, 0.42),
                              (T_SWEEP - 0.06, 0.62, 0.50)):
            if t0 <= t <= t0 + dur:
                q = (t - t0) / dur
                return ease_in_out_cubic(q), gain * math.sin(math.pi * q) ** 0.8
        return 0.0, 0.0

    def cells(self, g):
        """Centres/sizes of the three spec pills for the current split amount."""
        r = min(g["r"], g["h"] * 0.5)
        gap = lerp(-2.0 * r, 24.0, g["split"])
        wi = (g["w"] - 2.0 * gap) / 3.0
        return [(g["cx"] + k * (wi + gap), wi) for k in (-1, 0, 1)], gap

    # -------------------------------------------------------------- draw
    def draw(self, canvas, base, t):
        if t <= T_APPEAR:
            return
        g = self.geo(t)
        w, h = g["w"], g["h"]
        if w < 3 or h < 3:
            return
        cx, cy, r = g["cx"], g["cy"], g["r"]
        st = self.style(t)
        sweep_pos, sweep_gain = self._sweep(t)
        sweep = dict(pos=sweep_pos, gain=sweep_gain, width=88.0, angle=math.radians(24)) if sweep_gain > 0 else None
        p = lerp(2.0, 2.5, clamp(max(self.k_card(t), self.k_plate(t))))
        if g["split"] > 0.002:
            cells, gap = self.cells(g)
            kk = 30.0 * smooth(g["split"] * 2.5)

            def sdf(X, Y):
                d = None
                for ccx, wi in cells:
                    di = G.sd_rrect(X, Y, ccx, cy, wi, h, r)
                    d = di if d is None else G.smin(d, di, kk)
                return d
        else:
            def sdf(X, Y):
                return G.sd_rrect(X, Y, cx, cy, w, h, r, p)
        G.render_glass(canvas, base, sdf, (cx - w / 2 - 8, cy - h / 2 - 8, cx + w / 2 + 8, cy + h / 2 + 8), st,
                       box=(cy - h / 2, h), sweep=sweep, center=(cx, cy))
        self._content(canvas, t, g)

    def _content(self, canvas, t, g):
        cx, cy = g["cx"], g["cy"]

        def rise(t0, dur=0.30):
            e = ease_out_cubic(seg(t, t0, t0 + dur))
            return e, (1.0 - e) * 18.0, (1.0 - e) * 7.0          # opacity, y offset, blur

        # ---- intro title
        out = 1.0 - seg(t, fr(43), fr(45.5))
        if out > 0 and t < fr(45.6):
            ob = (1.0 - out) * 5.0
            o, dy, bl = rise(T_APPEAR + 0.20)
            kx = self.s_kdrop.w + 12.0
            x0 = cx - (self.s_kicker.adv + kx) / 2.0
            T.blit(canvas, self.s_kdrop, x0 - 2, cy - 113 - self.s_kdrop.h / 2 + dy - 1, o * out, blur=bl + ob)
            x, y = self.s_kicker.at(x0 + kx, cy - 113, "l")
            T.blit(canvas, self.s_kicker, x, y + dy, o * out, blur=bl + ob)
            o, dy, bl = rise(T_APPEAR + 0.27, 0.34)
            x, y = self.s_head.at(cx - 6, cy + 2, "c")
            T.blit(canvas, self.s_head, x, y + dy * 1.4, o * out, blur=bl * 1.6 + ob, scale=lerp(0.94, 1.0, o))
            o, dy, bl = rise(T_APPEAR + 0.37)
            x, y = self.s_sub.at(cx, cy + 122, "c")
            T.blit(canvas, self.s_sub, x, y + dy, o * out, blur=bl + ob)
            ol = ease_out_cubic(seg(t, T_APPEAR + 0.34, T_APPEAR + 0.70))
            T.blit(canvas, self.s_line, cx - self.s_line.w / 2, cy + 86, ol * out, scale=lerp(0.2, 1.0, ol))

        # ---- compact brand
        vis = 0.0
        for a, b in ((fr(51.5), fr(136.2)), (fr(177.5), fr(297.5))):
            vis = max(vis, ease_out_cubic(seg(t, a, a + 0.20)) * (1.0 - seg(t, b, b + 0.09)))
        if vis > 0.004:
            gap1, gap2 = 11.0, 15.0
            tot = self.s_bdrop.w + gap1 + self.s_b1.adv + gap2 + 2 + gap2 + self.s_b2.adv
            x = cx - tot / 2.0 - 2
            bl = (1.0 - vis) * 4.0
            T.blit(canvas, self.s_bdrop, x, cy - self.s_bdrop.h / 2 - 1, vis, blur=bl)
            x += self.s_bdrop.w + gap1
            px, py = self.s_b1.at(x, cy + 0.5, "l")
            T.blit(canvas, self.s_b1, px, py, vis, blur=bl)
            x += self.s_b1.adv + gap2
            T.blit(canvas, self.s_bdiv, x - 1, cy - self.s_bdiv.h / 2, vis, blur=bl)
            x += 2 + gap2
            px, py = self.s_b2.at(x, cy + 0.5, "l")
            T.blit(canvas, self.s_b2, px, py, vis, blur=bl)

        # ---- spec cells
        if T_SPEC < t < T_SPEC_END + 0.1:
            cells, _ = self.cells(g)
            out = 1.0 - seg(t, fr(170.5), fr(172.5))
            for k, ((ccx, _wi), (s_lab, s_val)) in enumerate(zip(cells, self.s_spec)):
                o, dy, bl = rise(T_SPEC + 0.13 + 0.065 * k, 0.26)
                x, y = s_lab.at(ccx, cy - 25, "c")
                T.blit(canvas, s_lab, x, y + dy * 0.6, o * out, blur=bl)
                x, y = s_val.at(ccx, cy + 13, "c")
                T.blit(canvas, s_val, x, y + dy, o * out, blur=bl)


# =========================================================================
#  FEATURE PILLS  (lower third: a round bubble that pours out a label)
# =========================================================================
class FeaturePill:
    X0, CY, R, HB = 60.0, 1332.0, 56.0, 84.0

    def __init__(self, icon, kicker, label, t_in, t_out):
        self.t_in, self.t_out = t_in, t_out
        self.s_icon = T.icon(icon, 58)
        self.s_kick = T.with_shadow(T.text(kicker, 15.5, 700, False, 0.30, T.GOLD), 4.0, 2.0, 0.5)
        self.s_lab = T.with_shadow(T.text(label, 31, 700, False, 0.085, T.WHITE), 6.0, 2.0, 0.5)
        self.bx = self.X0 + self.R
        self.tx = self.bx + self.R + 20.0
        self.right = self.tx + max(self.s_kick.adv, self.s_lab.adv) + 44.0
        self.pop = Track(0.0, [(t_in, 1.0, dict(freq=3.3, damp=0.56)),
                               (t_out + 0.17, 0.0, dict(dur=0.12, ease=ease_in_cubic))])
        self.ext = Track(0.0, [(t_in + 0.09, 1.0, dict(freq=2.7, damp=0.72)),
                               (t_out + 0.03, 0.0, dict(dur=0.17, ease=ease_in_cubic))])

    def active(self, t):
        return self.t_in < t < self.t_out + 0.32

    def travel(self, t0, t1):
        if not (self.active(t0) or self.active(t1)):
            return 0.0
        return max(abs(self.pop(t0) - self.pop(t1)) * self.R,
                   abs(self.ext(t0) - self.ext(t1)) * (self.right - self.bx))

    def draw(self, canvas, base, t):
        if not self.active(t):
            return
        pop, ext = self.pop(t), self.ext(t)
        if pop <= 0.02:
            return
        R = self.R * pop
        bx, cy = self.bx, self.CY
        right = bx + (self.right - bx) * max(ext, 0.0)
        hb = self.HB * min(1.0, pop)
        kk = 30.0 * min(1.0, pop)

        def sdf(X, Y):
            d = G.sd_circle(X, Y, bx, cy, R)
            if right - bx > 4.0:
                d = G.smin(d, G.sd_rrect(X, Y, (bx + right) / 2.0, cy, right - bx, hb, hb / 2.0), kk)
            return d

        G.render_glass(canvas, base, sdf, (bx - R - 6, cy - R - 6, max(right, bx + R) + 6, cy + R + 6), ST_PILL,
                       box=(cy - R, 2 * R))
        T.blit(canvas, self.s_icon, bx - self.s_icon.w / 2, cy - self.s_icon.h / 2, clamp(pop * 1.4 - 0.2),
               scale=max(0.05, pop))
        out = 1.0 - seg(t, self.t_out - 0.02, self.t_out + 0.07)
        for sp, yy, dl in ((self.s_kick, cy - 19, 0.0), (self.s_lab, cy + 13, 0.045)):
            e = ease_out_cubic(seg(t, self.t_in + 0.20 + dl, self.t_in + 0.44 + dl))
            if e * out <= 0.004:
                continue
            x, y = sp.at(self.tx, yy, "l")
            T.blit(canvas, sp, x - (1.0 - e) * 16.0, y, e * out, blur=(1.0 - e) * 5.0 + (1.0 - out) * 4.0)


# =========================================================================
#  THE WHOLE UI LAYER
# =========================================================================
class UI:
    def __init__(self):
        self.island = Island()
        self.pills = [FeaturePill(**f) for f in FEATURES]

    def travel(self, t0, t1):
        """Largest on-screen displacement (px) of any glass edge between two times."""
        return max([self.island.travel(t0, t1)] + [p.travel(t0, t1) for p in self.pills])

    def draw(self, canvas, base, t):
        for p in self.pills:
            p.draw(canvas, base, t)
        self.island.draw(canvas, base, t)

    def render(self, base, i):
        """UI for output frame i.  While the glass moves, the frame is integrated over
        the open shutter with enough sub-samples that edges smear instead of strobing."""
        t0 = i / FPS
        t1 = (i + UI_SHUTTER) / FPS
        d = self.travel(t0, t1)
        if d < 0.35:
            canvas = base.copy()
            self.draw(canvas, base, (i + UI_SHUTTER * 0.5) / FPS)
            return canvas
        n = int(min(UI_MAX_SAMPLES, max(3, math.ceil(d / 2.2))))
        acc = np.zeros_like(base)
        canvas = np.empty_like(base)
        for j in range(n):
            np.copyto(canvas, base)
            self.draw(canvas, base, (i + UI_SHUTTER * (j + 0.5) / n) / FPS)
            acc += canvas
        acc /= n
        return acc


# =========================================================================
#  FRAME PIPELINE
# =========================================================================
class Film:
    def __init__(self, raw):
        self.raw = raw                       # (N, 1280, 720, 3) uint8
        self.ui = UI()
        self.grain = L.Grain((W, H))
        self.snow = SN.Snow(W, H)
        self.cars = CR.Cars(raw)             # petrol sedans replace the electric cars
        self.repairs = FX.Repairs(raw)
        self._hero = None

    def plate(self, i, src_index=None, cam=True):
        """Graded background plate for output frame i (camera move, sharpen, grade, bloom)."""
        idx = i if src_index is None else src_index
        src = np.asarray(self.raw[idx])
        if self.cars.active(idx):
            src = self.cars.apply(np.array(src, dtype=np.uint8), idx)
        if self.repairs.active(idx):
            src = self.repairs.apply(np.array(src, dtype=np.uint8), idx)
        if cam:
            img = L.camera(src, (W, H), lambda u: cam_params(i, u), shutter=SHUTTER)
            moving = img.dtype != np.uint8
        else:
            img = L.warp(src, (W, H))
            moving = False
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img / 255.0
        if not moving:
            img = L.unsharp(img, 1.1, 0.42)
        img = L.grade(img, contrast=0.10)
        img = L.bloom(img)
        return img

    def base(self, i):
        img = self.plate(i)
        if WIPE_F0 <= i < WIPE_F1:
            if self._hero is None:
                self._hero = self.plate(137, cam=False)
            img = wave_wipe(img, self._hero, i)
        img = L.flash(img, flash_amount(i))
        self.snow.draw(img, i / FPS, SNOW[shot_of(i)])
        img = L.vignette(img, 0.16)
        return img

    def frame(self, i, ui=True, grain=True):
        base = self.base(i)
        out = self.ui.render(base, i) if ui else base
        k = 1.0 - ease_in_out_cubic(seg(i / FPS, fr(350), fr(360.5)))      # final fade
        if k < 1.0:
            out = out * k
        if grain:
            out = self.grain.apply(out, i, GRAIN)
        return L.to_u8(out, i)
