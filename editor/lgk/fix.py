"""Repair the two brand words the source renders with AI spelling errors.

Both sit on surfaces that move with the camera, so nothing can be painted at a
fixed screen position: the shot is tracked with SIFT features inside a region of
interest, and the fix (erase mask + correctly spelled word) is drawn in the
reference frame and warped onto every frame with that transform.

  * shot 3 - the gantry sign reads "PAMR"        -> "PAMIR"
  * shot 7 - the tachometer dial reads "PAMRI"   -> "PAMIR"

The erase is a TELEA inpaint of the old lettering only, so the surface (banner
panel / dial face) keeps its own gradient; the new word is drawn at 4x, blurred
by the local motion of the shot and measured against the source softness, then
composited under the same transform.
"""
import math

import cv2
import numpy as np

from . import typo as T

_SIFT = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.006)
_BF = cv2.BFMatcher()


def _prep(gray):
    return cv2.createCLAHE(3.0, (8, 8)).apply(gray)


class Tracker:
    """Similarity transform (reference frame -> frame n) from SIFT feature matches.

    Only features inside a mask around the element are used, and the mask is
    carried along with the accumulated transform: nearby parts of the frame move
    with a different parallax, and letting them vote would drag the patch off the
    letters it is supposed to sit on.
    """

    def __init__(self, raw, ref, roi, mask=None):
        self.raw, self.ref, self.roi = raw, ref, roi
        self.mask = mask                            # ("circle", cx, cy, r) or ("poly", pts)
        self.p0, self.d0 = self._features(ref, np.array([[1, 0, 0], [0, 1, 0]], np.float32))

    def _mask(self, M):
        if self.mask is None:
            return None
        y0, y1, x0, x1 = self.roi
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        if self.mask[0] == "circle":
            _, cx, cy, r = self.mask
            sc = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
            p = M @ np.array([cx, cy, 1.0], np.float32)
            cv2.circle(m, (int(round(p[0] - x0)), int(round(p[1] - y0))), int(round(r * sc)), 255, -1)
        else:
            pts = np.float32(self.mask[1]).reshape(-1, 1, 2)
            q = cv2.transform(pts, M).reshape(-1, 2)
            q[:, 0] -= x0
            q[:, 1] -= y0
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 255)
        return m

    def _features(self, n, M):
        y0, y1, x0, x1 = self.roi
        g = cv2.cvtColor(np.asarray(self.raw[n, y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
        m = self._mask(M)
        kp, ds = _SIFT.detectAndCompute(_prep(g), m)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 2) if kp else np.zeros((0, 2), np.float32)
        return pts, ds

    def _match(self, a, b, Ma, Mb):
        """Partial affine mapping frame a -> frame b."""
        pa, da = self._features(a, Ma)
        pb, db = self._features(b, Mb)
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return None
        m = _BF.knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
        if len(good) < 8:
            return None
        src = np.float32([pa[g.queryIdx] for g in good]).reshape(-1, 1, 2)
        dst = np.float32([pb[g.trainIdx] for g in good]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                             maxIters=8000, confidence=0.9995)
        if M is None or inl is None or int(inl.sum()) < 6:
            return None
        return M.astype(np.float32)

    def refine(self, M, n, box, pad=12):
        """Lock the chained estimate onto the lettering with a translation-only NCC.

        The SIFT chain drifts a pixel or so per hop; the words are the strongest
        local feature there is, so matching the reference template - resampled to
        the predicted scale/rotation - around the predicted spot corrects the drift.
        """
        x0, y0, x1, y1 = box
        tpl = cv2.cvtColor(np.asarray(self.raw[self.ref, y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
        sc = math.sqrt(max(1e-6, M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        tpl = cv2.warpAffine(cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC),
                             cv2.getRotationMatrix2D((0, 0), ang, 1.0),
                             (max(4, int(round((x1 - x0) * sc))), max(4, int(round((y1 - y0) * sc)))),
                             flags=cv2.INTER_CUBIC)
        img = cv2.cvtColor(np.asarray(self.raw[n]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        th, tw = tpl.shape
        pc = M @ np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, 1.0], np.float32)
        w0, h0 = int(round(pc[0] - tw / 2.0)) - pad, int(round(pc[1] - th / 2.0)) - pad
        w1, h1 = w0 + tw + 2 * pad, h0 + th + 2 * pad
        if w0 < 0 or h0 < 0 or w1 > img.shape[1] or h1 > img.shape[0]:
            return M
        r = cv2.matchTemplate(img[h0:h1, w0:w1], tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, ml = cv2.minMaxLoc(r)
        if mx < 0.20:
            return M
        dx = (w0 + ml[0] + tw / 2.0) - pc[0]
        dy = (h0 + ml[1] + th / 2.0) - pc[1]
        M2 = M.copy()
        M2[0, 2] += dx
        M2[1, 2] += dy
        return M2

    def locate(self, frames, box, radii=(70, 110, 160, 240)):
        """Reference -> n by matching the reference template itself on every frame.

        A transform chain drifts while a camera move is accelerating; re-anchoring
        on the actual lettering every frame cannot, so each frame is searched for
        the reference template over a small ladder of scales and rotations, seeded
        by the previous frame's answer.
        """
        x0, y0, x1, y1 = box
        tpl0 = cv2.cvtColor(np.asarray(self.raw[self.ref, y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
        tpl0 = tpl0.astype(np.float32)
        th0, tw0 = tpl0.shape
        cref = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], np.float32)
        ident = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        out = {self.ref: ident}
        imgs = {}

        def gray(n):
            if n not in imgs:
                imgs[n] = cv2.cvtColor(np.asarray(self.raw[n]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            return imgs[n]

        for step in (1, -1):
            M = ident.copy()
            n = self.ref
            prev_v = 0.0
            while (n + step) in set(frames):
                nxt = n + step
                pc = M @ np.array([cref[0], cref[1], 1.0], np.float32)
                sc0 = math.sqrt(max(1e-6, M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
                ang0 = math.degrees(math.atan2(M[1, 0], M[0, 0]))
                rad = int(min(radii[-1], max(radii[0], prev_v * 3.0 + 55.0)))
                img = gray(nxt)
                best = (-2.0, M.copy())
                for sc in (sc0 * f for f in (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06)):
                    for ang in (ang0 + a for a in (-5.0, -2.5, 0.0, 2.5, 5.0)):
                        R = cv2.getRotationMatrix2D((0, 0), ang, sc)
                        tw, th = max(8, int(round(tw0 * sc))), max(8, int(round(th0 * sc)))
                        t = cv2.warpAffine(tpl0, np.array([[R[0, 0], R[0, 1], 0], [R[1, 0], R[1, 1], 0]], np.float32),
                                           (tw, th), flags=cv2.INTER_LINEAR)
                        w0 = int(round(pc[0] - tw / 2.0)) - rad
                        h0 = int(round(pc[1] - th / 2.0)) - rad
                        w1, h1 = w0 + tw + 2 * rad, h0 + th + 2 * rad
                        if w0 < 0 or h0 < 0 or w1 > img.shape[1] or h1 > img.shape[0]:
                            continue
                        r = cv2.matchTemplate(img[h0:h1, w0:w1], t, cv2.TM_CCOEFF_NORMED)
                        _mn, mx, _ml, ml = cv2.minMaxLoc(r)
                        if mx > best[0]:
                            px = w0 + ml[0] + tw / 2.0
                            py = h0 + ml[1] + th / 2.0
                            M2 = np.array([[R[0, 0], R[0, 1], 0.0], [R[1, 0], R[1, 1], 0.0]], np.float32)
                            M2[0, 2] = px - (M2[0, 0] * cref[0] + M2[0, 1] * cref[1])
                            M2[1, 2] = py - (M2[1, 0] * cref[0] + M2[1, 1] * cref[1])
                            best = (mx, M2)
                p_new = best[1] @ np.array([cref[0], cref[1], 1.0], np.float32)
                prev_v = float(np.hypot(*(p_new[:2] - pc[:2])))
                M = best[1]
                out[nxt] = M
                n = nxt
        return out

    def all(self, frames, box=None):
        """Reference -> n for every frame, composed outwards from the reference.

        Neighbouring frames share almost all of their pixels, which survives the
        motion blur of a camera move far better than one long jump. Each hop is
        corrected against the reference template before it becomes the base of the
        next one, so the drift never has a chance to build up.
        """
        frames = sorted(frames)
        fs = set(frames)
        ident = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        out = {self.ref: ident}
        for step in (1, -1):
            M = ident
            n = self.ref
            while (n + step) in fs:
                nxt = n + step
                E = self._match(n, nxt, M, M)
                M2 = M
                if E is not None:
                    M2 = (np.vstack([E, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])).astype(np.float32)[:2]
                if box is not None:
                    M2 = self.refine(M2, nxt, box)
                M = M2
                out[nxt] = M.copy()
                n = nxt
        return out


def _line_kernel(angle, length):
    """Normalised linear motion-blur kernel."""
    k = max(2, int(round(length)))
    if k <= 2:
        return None
    pad = k + 2
    img = np.zeros((pad, pad), np.float32)
    c = (pad - 1) / 2.0
    dx, dy = math.cos(angle), math.sin(angle)
    cv2.line(img, (int(round(c - dx * k / 2)), int(round(c - dy * k / 2))),
             (int(round(c + dx * k / 2)), int(round(c + dy * k / 2))), 1.0, 1)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    if img.sum() <= 0:
        return None
    return img / img.sum()


class WordFix:
    """Erase one word on the plate and draw a correctly spelled one in its place."""

    def __init__(self, frames, ref, roi, erase, text, left, cap_cy, cap_h, fit_w=None,
                 weight=800, italic=True, fill=(0.93, 0.94, 0.96), blur=0.9, thr=0.42, dil=5,
                 protect=False):
        self.frames = list(frames)
        self.ref = ref
        self.erased = erase                       # (x0, y0, x1, y1) in reference coords
        self.left, self.cap_cy, self.cap_h = float(left), float(cap_cy), float(cap_h)
        self.blur = blur
        self.thr, self.dil = float(thr), int(dil)
        self.protect = bool(protect)
        self._prot = None
        sp = T.text(text, cap_h / 0.70, weight, italic, 0.0, fill, ss=4)
        rgba = sp.rgba
        sx = 1.0
        if fit_w:                                  # squeeze the word to the width of the old one
            target = float(fit_w) * rgba.shape[1] / max(sp.adv, 1e-3)
            sx = target / rgba.shape[1]
            rgba = cv2.resize(rgba, (max(1, int(round(target))), rgba.shape[0]),
                              interpolation=cv2.INTER_AREA)
        self.sprite = np.ascontiguousarray(rgba, np.float32)
        # sprite top-left in reference coords, so that ink starts at `left`, cap centred on cap_cy
        self.spr_xy = (left - sp.left * sx, cap_cy - 0.5 * sp.cap_h - sp.cap_top)

    # ------------------------------------------------------------------ helpers
    def _warp(self, src, M, origin, pad=2):
        h, w = src.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) + np.float32(origin)
        q = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
        x0, y0 = int(math.floor(q[:, 0].min())) - pad, int(math.floor(q[:, 1].min())) - pad
        x1, y1 = int(math.ceil(q[:, 0].max())) + pad, int(math.ceil(q[:, 1].max())) + pad
        T_ = np.array([[1, 0, -x0], [0, 1, -y0]], np.float32)
        A = np.array([[1, 0, origin[0]], [0, 1, origin[1]]], np.float32)
        Mp = (T_ @ (np.vstack([M, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]])))[:2].astype(np.float32)
        ch = src.shape[2] if src.ndim == 3 else 1
        out = cv2.warpAffine(src, Mp, (max(1, x1 - x0), max(1, y1 - y0)), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, (x0, y0, x1, y1)

    # --------------------------------------------------------------------- draw
    def apply(self, img, M, alpha=1.0, vel=None):
        """Patch `img` (uint8 RGB, source resolution) in place. M maps reference -> this frame."""
        if M is None or alpha <= 0.0:
            return img
        H, W = img.shape[:2]
        # 1) erase the wrong lettering
        x0, y0, x1, y1 = self.erased
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        mask[:] = 255
        wm, (mx0, my0, mx1, my1) = self._warp(mask, M, (x0, y0), pad=12)
        rx0, ry0 = max(0, mx0), max(0, my0)
        rx1, ry1 = min(W, mx1), min(H, my1)
        if rx1 <= rx0 or ry1 <= ry0:
            return img
        region = np.ascontiguousarray(img[ry0:ry1, rx0:rx1])
        sel = wm[ry0 - my0:ry1 - my0, rx0 - mx0:rx1 - mx0] > 96
        # the letters are the bright pixels inside the box; masking just those keeps the
        # surface gradient (banner panel / dial face) instead of flooding the whole box
        lum = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        vals = lum[sel] if sel.any() else lum.ravel()
        p60, p99 = np.percentile(vals, 60), np.percentile(vals, 99)
        th = max(0.14, p60 + self.thr * max(p99 - p60, 0.05))
        # white lettering only: the orange needle and the gold badge are saturated and
        # must survive the erase untouched
        mreg = ((lum > th) & (sat < 0.32) & sel).astype(np.uint8) * 255
        mreg = cv2.dilate(mreg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dil, self.dil)))
        self._prot = None
        if self.protect:
            pm = ((sat > 0.42) & (lum > 0.22) & sel).astype(np.uint8) * 255
            pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dil + 3, self.dil + 3)))
            pm = cv2.GaussianBlur(pm.astype(np.float32) / 255.0, (0, 0), 1.6)
            self._prot = (pm, (rx0, ry0))
        img[ry0:ry1, rx0:rx1] = cv2.inpaint(region, mreg, 4, cv2.INPAINT_TELEA)
        # 2) draw the right word
        spr = self.sprite
        if self.blur > 0.05:
            spr = cv2.GaussianBlur(spr, (0, 0), self.blur)
        warped, (sx0, sy0, sx1, sy1) = self._warp(spr, M, self.spr_xy, pad=4)
        if vel is not None and float(np.hypot(*vel)) > 1.6:
            k = _line_kernel(math.atan2(vel[1], vel[0]), 0.55 * float(np.hypot(*vel)))
            if k is not None:
                warped = cv2.filter2D(warped, -1, k)
        a = warped[..., 3:4] * alpha
        rgb = warped[..., :3]
        qx0, qy0 = max(0, sx0), max(0, sy0)
        qx1, qy1 = min(W, sx1), min(H, sy1)
        if qx1 <= qx0 or qy1 <= qy0:
            return img
        a = a[qy0 - sy0:qy1 - sy0, qx0 - sx0:qx1 - sx0].copy()
        rgb = rgb[qy0 - sy0:qy1 - sy0, qx0 - sx0:qx1 - sx0].copy()
        if self._prot is not None:                       # keep whatever sits in front of the word
            pm, (px, py) = self._prot
            sub = np.ones(a.shape[:2], np.float32)
            ax0, ay0 = max(0, px - qx0), max(0, py - qy0)
            bx0, by0 = max(0, qx0 - px), max(0, qy0 - py)
            hh = min(a.shape[0] - ax0, pm.shape[0] - by0)
            ww = min(a.shape[1] - ay0, pm.shape[1] - bx0)
            if hh > 0 and ww > 0:
                sub[ax0:ax0 + hh, ay0:ay0 + ww] -= pm[by0:by0 + hh, bx0:bx0 + ww]
            a = a * np.clip(sub, 0.0, 1.0)[..., None]
        dst = img[qy0:qy1, qx0:qx1].astype(np.float32) / 255.0
        dst = dst * (1.0 - a) + rgb
        img[qy0:qy1, qx0:qx1] = np.clip(dst * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return img


# =========================================================================
#  WHAT GETS REPAIRED IN THIS EDIT
# =========================================================================
# The tachometer dial reads "PAMRI" for a little over a second while the camera
# pushes in on it - large enough to read, so it is replaced with "PAMIR".
# (The gantry sign in shot 3 also misspells the brand, but there the word is only
# six pixels tall in the source: it resolves as a texture, not as letters, and
# repainting it would only add artefacts.)
DIAL_SHOT = dict(ref=232, frames=range(224, 250), roi=(330, 820, 170, 660),
                 mask=("circle", 400, 640, 300), box=(336, 497, 424, 531))


class Repairs:
    """Per-frame text repairs for the whole film, tracked lazily."""

    def __init__(self, raw):
        self.raw = raw
        self._word = WordFix(frames=DIAL_SHOT["frames"], ref=DIAL_SHOT["ref"], roi=DIAL_SHOT["roi"],
                             erase=(335, 497, 425, 531), text="PAMIR", left=340.0, cap_cy=514.0,
                             cap_h=25.0, fit_w=79.0, weight=700, italic=True,
                             fill=(0.87, 0.88, 0.90), blur=1.2, thr=0.40, dil=4, protect=True)
        self._tracker = None
        self._M = None

    def active(self, i):
        return i in DIAL_SHOT["frames"]

    def apply(self, img, i):
        if not self.active(i):
            return img
        if self._M is None:
            self._tracker = Tracker(self.raw, DIAL_SHOT["ref"], DIAL_SHOT["roi"], mask=DIAL_SHOT["mask"])
            self._M = self._tracker.locate(DIAL_SHOT["frames"], DIAL_SHOT["box"])
        M = self._M.get(i)
        if M is None:
            return img
        self._word.apply(img, M)
        return img
