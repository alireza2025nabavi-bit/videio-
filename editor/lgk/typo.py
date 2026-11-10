"""Typography + icon sprites (Pillow) and sub-pixel compositing (OpenCV)."""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

GOLD_GRAD = [(0.0, (1.00, 0.925, 0.66)), (0.48, (0.975, 0.78, 0.30)), (1.0, (0.86, 0.60, 0.13))]
GOLD = (0.965, 0.785, 0.32)
WHITE = (1.0, 1.0, 1.0)


def font_path(weight=600, italic=False):
    return os.path.join(FONT_DIR, f"Montserrat-{weight}{'Italic' if italic else ''}.ttf")


# ------------------------------------------------------------------ kerning
class Kerner:
    """Pair kerning read straight from the font's GPOS table.

    Pillow's basic layout engine ignores GPOS, so we apply it ourselves.
    """
    _cache = {}

    def __new__(cls, path):
        if path in cls._cache:
            return cls._cache[path]
        self = super().__new__(cls)
        cls._cache[path] = self
        tt = TTFont(path)
        self.upem = tt["head"].unitsPerEm
        self.cmap = tt.getBestCmap()
        self.subtables = []
        if "GPOS" in tt:
            gpos = tt["GPOS"].table
            idx = set()
            for fr in gpos.FeatureList.FeatureRecord:
                if fr.FeatureTag == "kern":
                    idx.update(fr.Feature.LookupListIndex)
            for li in sorted(idx):
                lk = gpos.LookupList.Lookup[li]
                group = []
                for st in lk.SubTable:
                    if lk.LookupType == 9:
                        st = st.ExtSubTable
                    if getattr(st, "LookupType", lk.LookupType) != 2 and lk.LookupType != 2 and not hasattr(st, "PairSet") and not hasattr(st, "Class1Record"):
                        continue
                    if hasattr(st, "PairSet"):
                        table = {}
                        for g1, ps in zip(st.Coverage.glyphs, st.PairSet):
                            for pvr in ps.PairValueRecord:
                                v = pvr.Value1.XAdvance if (pvr.Value1 is not None and hasattr(pvr.Value1, "XAdvance")) else 0
                                table[(g1, pvr.SecondGlyph)] = v
                        group.append(("pairs", set(st.Coverage.glyphs), table))
                    elif hasattr(st, "Class1Record"):
                        group.append(("classes", set(st.Coverage.glyphs), st.ClassDef1.classDefs,
                                      st.ClassDef2.classDefs, st.Class1Record))
                if group:
                    self.subtables.append(group)
        return self

    def pair(self, a, b):
        """Kerning between characters a and b in em units."""
        g1 = self.cmap.get(ord(a))
        g2 = self.cmap.get(ord(b))
        if g1 is None or g2 is None:
            return 0.0
        total = 0.0
        for group in self.subtables:
            for sub in group:
                if g1 not in sub[1]:
                    continue
                if sub[0] == "pairs":
                    if (g1, g2) in sub[2]:
                        total += sub[2][(g1, g2)]
                        break
                else:
                    _, _, cd1, cd2, recs = sub
                    c1 = cd1.get(g1, 0)
                    c2 = cd2.get(g2, 0)
                    v1 = recs[c1].Class2Record[c2].Value1
                    v = v1.XAdvance if (v1 is not None and hasattr(v1, "XAdvance")) else 0
                    total += v
                    break
        return total / self.upem


# ------------------------------------------------------------------ sprites
class Sprite:
    """Premultiplied RGBA float sprite with typographic anchor info."""

    def __init__(self, rgba, left=0.0, adv=None, cap_top=0.0, cap_bot=None):
        self.rgba = np.ascontiguousarray(rgba, dtype=np.float32)
        self.h, self.w = rgba.shape[:2]
        self.left = left                                   # x of the layout origin
        self.adv = float(self.w if adv is None else adv)   # advance width
        self.cap_top = cap_top
        self.cap_bot = float(self.h if cap_bot is None else cap_bot)

    @property
    def cap_h(self):
        return self.cap_bot - self.cap_top

    def at(self, x, cy, anchor="l"):
        """Top-left blit position so the text sits at x (l/c/r) and is optically centred on cy."""
        top = cy - 0.5 * (self.cap_top + self.cap_bot)
        if anchor == "c":
            return x - self.adv * 0.5 - self.left, top
        if anchor == "r":
            return x - self.adv - self.left, top
        return x - self.left, top


def _fill(mask, fill, y0, y1):
    """Colour an alpha mask with a flat colour or a vertical gradient -> premultiplied RGBA."""
    h, w = mask.shape
    if isinstance(fill, list):
        ys = (np.arange(h, dtype=np.float32) + 0.5 - y0) / max(y1 - y0, 1.0)
        ys = np.clip(ys, 0, 1)
        stops = np.array([p for p, _ in fill], np.float32)
        cols = np.array([c for _, c in fill], np.float32)
        col = np.stack([np.interp(ys, stops, cols[:, i]) for i in range(3)], axis=-1)[:, None, :]
        col = np.broadcast_to(col, (h, w, 3))
    else:
        col = np.broadcast_to(np.asarray(fill, np.float32), (h, w, 3))
    rgba = np.empty((h, w, 4), np.float32)
    rgba[..., :3] = col * mask[..., None]
    rgba[..., 3] = mask
    return rgba


_text_cache = {}


def text(txt, size, weight=600, italic=False, tracking=0.0, fill=WHITE, opacity=1.0, ss=3):
    """Render a line of text. size in px, tracking in em."""
    key = (txt, size, weight, italic, tracking, str(fill), opacity, ss)
    if key in _text_cache:
        return _text_cache[key]
    path = font_path(weight, italic)
    f = ImageFont.truetype(path, int(round(size * ss)))
    kern = Kerner(path)
    px = size * ss
    xs, x = [], 0.0
    for i, ch in enumerate(txt):
        if i:
            x += kern.pair(txt[i - 1], ch) * px + tracking * px
        xs.append(x)
        x += f.getlength(ch)
    total = x
    asc, desc = f.getmetrics()
    pad = int(px * 0.30) + 4
    Wd = int(math.ceil(total)) + 2 * pad
    Hd = asc + desc + 2 * pad
    Wd += (-Wd) % ss
    Hd += (-Hd) % ss
    img = Image.new("L", (Wd, Hd), 0)
    dr = ImageDraw.Draw(img)
    for ch, cx in zip(txt, xs):
        dr.text((pad + cx, pad), ch, font=f, fill=255)
    mask = np.asarray(img, np.float32) / 255.0
    mask = cv2.resize(mask, (Wd // ss, Hd // ss), interpolation=cv2.INTER_AREA)
    hb = f.getbbox("H")
    cap_top = (pad + hb[1]) / ss
    cap_bot = (pad + hb[3]) / ss
    rgba = _fill(mask, fill, cap_top, cap_bot) * opacity
    sp = Sprite(rgba, left=pad / ss, adv=total / ss, cap_top=cap_top, cap_bot=cap_bot)
    _text_cache[key] = sp
    return sp


def with_shadow(sp, blur=6.0, dy=2.0, alpha=0.45):
    """Bake a soft drop shadow under a sprite (keeps anchors)."""
    pad = int(math.ceil(blur * 3 + abs(dy))) + 1
    big = cv2.copyMakeBorder(sp.rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = big[..., 3]
    sh = cv2.GaussianBlur(a, (0, 0), blur)
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    sh = cv2.warpAffine(sh, M, (sh.shape[1], sh.shape[0])) * alpha
    out = np.empty_like(big)
    out[..., :3] = big[..., :3]                        # shadow is black: adds nothing to rgb
    out[..., 3] = a + sh * (1.0 - a)
    return Sprite(out, left=sp.left + pad, adv=sp.adv, cap_top=sp.cap_top + pad, cap_bot=sp.cap_bot + pad)


# -------------------------------------------------------------------- icons
def _poly_sprite(draw_fn, size, fill, ss=8):
    S = int(size * ss)
    img = Image.new("L", (S, S), 0)
    draw_fn(ImageDraw.Draw(img), S)
    mask = cv2.resize(np.asarray(img, np.float32) / 255.0, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    return Sprite(_fill(mask, fill, size * 0.12, size * 0.88))


_icon_cache = {}


def icon(name, size, fill=None):
    fill = GOLD_GRAD if fill is None else fill
    key = (name, size, str(fill))
    if key in _icon_cache:
        return _icon_cache[key]

    def drop(d, S):
        # oil drop: circle + tangent cone
        cx, cy, r = S * 0.5, S * 0.60, S * 0.285
        tip = (cx, S * 0.07)
        dist = cy - tip[1]
        ang = math.acos(r / dist)
        p1 = (cx - r * math.sin(ang), cy - r * math.cos(ang))
        p2 = (cx + r * math.sin(ang), cy - r * math.cos(ang))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        d.polygon([tip, p1, p2], fill=255)
        # crescent highlight cut-out
        rr = r * 0.60
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=95, end=170, fill=0, width=max(1, int(S * 0.045)))

    def chevrons(d, S):
        wdt = max(1, int(S * 0.115))
        for k, x0 in enumerate((0.20, 0.50)):
            pts = [(S * x0, S * 0.24), (S * (x0 + 0.26), S * 0.5), (S * x0, S * 0.76)]
            d.line(pts, fill=255, width=wdt, joint="curve")
            for p in pts:
                d.ellipse([p[0] - wdt / 2, p[1] - wdt / 2, p[0] + wdt / 2, p[1] + wdt / 2], fill=255)

    def shield(d, S):
        pts = []
        top, bot = 0.12, 0.92
        # outline of a shield (right half mirrored)
        half = [(0.0, top), (0.34, top + 0.10), (0.34, 0.50), (0.26, 0.70), (0.13, 0.84), (0.0, bot)]
        for x, y in half:
            pts.append((S * (0.5 + x), S * y))
        for x, y in reversed(half):
            pts.append((S * (0.5 - x), S * y))
        d.polygon(pts, fill=255)
        wdt = max(1, int(S * 0.085))
        chk = [(S * 0.355, S * 0.50), (S * 0.465, S * 0.61), (S * 0.655, S * 0.385)]
        d.line(chk, fill=0, width=wdt, joint="curve")
        for p in chk:
            d.ellipse([p[0] - wdt / 2, p[1] - wdt / 2, p[0] + wdt / 2, p[1] + wdt / 2], fill=0)

    def dot(d, S):
        d.ellipse([S * 0.2, S * 0.2, S * 0.8, S * 0.8], fill=255)

    fn = {"drop": drop, "chevrons": chevrons, "shield": shield, "dot": dot}[name]
    sp = _poly_sprite(fn, size, fill)
    _icon_cache[key] = sp
    return sp


def bar(w, h, fill=WHITE, opacity=1.0, radius=None):
    """Small rounded bar (dividers, underlines)."""
    ss = 4
    img = Image.new("L", (int(w * ss) + 2 * ss, int(h * ss) + 2 * ss), 0)
    d = ImageDraw.Draw(img)
    r = (h * ss / 2) if radius is None else radius * ss
    d.rounded_rectangle([ss, ss, ss + w * ss, ss + h * ss], radius=r, fill=255)
    mask = cv2.resize(np.asarray(img, np.float32) / 255.0, (int(w) + 2, int(h) + 2), interpolation=cv2.INTER_AREA)
    return Sprite(_fill(mask, fill, 0, h + 2) * opacity)


# --------------------------------------------------------------- compositing
def blit(dst, sp, x, y, opacity=1.0, scale=1.0, blur=0.0, pivot=None):
    """Alpha-composite a sprite at a sub-pixel position (x, y = top-left of the unscaled sprite)."""
    if opacity <= 0.004 or scale <= 0.01:
        return
    H, W = dst.shape[:2]
    px, py = pivot if pivot is not None else (sp.w * 0.5, sp.h * 0.5)
    left = x + px - px * scale
    top = y + py - py * scale
    pad = int(math.ceil(blur * 3)) + 2
    X0 = int(math.floor(left)) - pad
    Y0 = int(math.floor(top)) - pad
    X1 = int(math.ceil(left + sp.w * scale)) + pad
    Y1 = int(math.ceil(top + sp.h * scale)) + pad
    if X1 <= 0 or Y1 <= 0 or X0 >= W or Y0 >= H:
        return
    M = np.float32([[scale, 0, left - X0 + 0.5 * scale - 0.5],
                    [0, scale, top - Y0 + 0.5 * scale - 0.5]])
    patch = cv2.warpAffine(sp.rgba, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if blur > 0.25:
        patch = cv2.GaussianBlur(patch, (0, 0), blur)
    cx0, cy0, cx1, cy1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
    patch = patch[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0]
    roi = dst[cy0:cy1, cx0:cx1]
    a = patch[..., 3:4] * opacity
    roi *= (1.0 - a)
    roi += patch[..., :3] * opacity
